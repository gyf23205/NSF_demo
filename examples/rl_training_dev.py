import sys
from pathlib import Path
my_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(my_path)
LOG_DIR = Path(my_path) / "examples" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)  # ensure it exists

import random
import numpy as np
import csv, json
import math
import time
from collections import defaultdict, deque
from time import strftime
from main_gui_ltl import compute_utilization
from ltl_core.binding_manager import *
from ltl_core.specification import *
from ltl_core.workspace import *
from ltl_core.labeler import *
from ltl_core.visualization import *
from ltl_core.allocator import *
from ltl_core.simulation import *


SLIDING_WINDOW = 60.0
TIME_OUT = 3000.0
grid_size = (50, 40)

# Event setup
FIREMSG_TIMES = [30.0, 45.0, 80.0, 110.0, 150.0, 250.0, 300.0]
SURVIVORMSG_TIMES = [25.0, 55.0, 70.0, 100.0, 156.0, 169.3, 264.5]
ATMMSG_TIMES = [35.0, 65.0, 90.0, 120.0, 154.2, 185.3, 210.4, 290.0]
# FIREMSG_TIMES = [30.0]
# SURVIVORMSG_TIMES = [25.0]
# ATMMSG_TIMES = [35.0]


class MetricsRecorder:
    """Episode recorder -> single unified CSV with step/ap_event/agent/episode rows."""
    def __init__(self, out_dir, window=60.0, human_overload_thresh=80, also_write_separate=False):
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.also_write_separate = also_write_separate

        # time-series (step rows)
        self.ts_rows = []

        # AP lifecycle
        self.ap_unlock_t = {}         # ap -> time unlocked
        self.ap_first_assign_t = {}   # ap -> time first assigned
        self.ap_complete_t = {}       # ap -> time completed
        self.ap_assigned_to = {}      # ap -> agent label

        # per-agent stats
        self.agent_switches = defaultdict(int)  # symbolic-task switches
        self.agent_prev_task = {}

        # per-human SA proxy (decay + credit)
        self.human_SA = defaultdict(float)
        self.SA_decay = 0.98
        self.SA_credit = 1.0

        # per-agent distance
        self.agent_prev_pos = {}
        self.agent_dist = defaultdict(float)

        # human overload / idle tracking
        self.human_overload_thresh = human_overload_thresh  # utilization is in percent (0..100)
        self.human_overload_steps  = defaultdict(int)       # label -> count
        self.human_idle_steps      = defaultdict(int)       # label -> count
        self.human_total_steps     = defaultdict(int)       # label -> count

        # timing
        self.mission_completed_t = None
        self.t0 = None
        self.t_last = 0.0

    def start_episode(self, t0=0.0, agents=None):
        self.t0 = t0
        self.t_last = t0
        if agents:
            for a in agents:
                self.agent_prev_pos[a.label] = tuple(a.pos)

    def note_unlocks(self, unlocked, t):
        for ap in unlocked:
            if ap not in self.ap_unlock_t:
                self.ap_unlock_t[ap] = t

    def note_assignments(self, assignments, t):
        # assignments: {agent_obj: ap}
        for agent, ap in assignments.items():
            if ap not in self.ap_first_assign_t:
                self.ap_first_assign_t[ap] = t
                self.ap_assigned_to[ap] = agent.label
            # count symbolic task switches
            cur = getattr(agent, "current_symbolic_task", None)
            prev = self.agent_prev_task.get(agent.label)
            if cur is not None and prev is not None and cur != prev:
                self.agent_switches[agent.label] += 1
            self.agent_prev_task[agent.label] = cur

    def note_completions(self, completed, t, agents_by_type):
        completed_set = set(completed)

        # Credit +1.0 to the recorded human assignee for each completed AP
        human_labels = {h.label for h in agents_by_type.get("humans", [])}
        for ap in completed_set:
            assignee = self.ap_assigned_to.get(ap)
            if assignee in human_labels:
                self.human_SA[assignee] += 1.0  # add exactly 1.0

        # Mark completion time (first time only)
        for ap in completed_set:
            if ap not in self.ap_complete_t:
                self.ap_complete_t[ap] = t

    def update_distances(self, agents):
        for a in agents:
            pl = self.agent_prev_pos.get(a.label)
            if pl is not None:
                dx = float(a.pos[0] - pl[0]); dy = float(a.pos[1] - pl[1])
                self.agent_dist[a.label] += math.hypot(dx, dy)
            self.agent_prev_pos[a.label] = tuple(a.pos)

    def decay_SA(self, agents_by_type):
        for h in agents_by_type.get("humans", []):
            self.human_SA[h.label] *= self.SA_decay

    def record_step(self, t, assignments, unlocked, completed, agents_by_type):
        # per-human overload/idle counters
        for h in agents_by_type.get("humans", []):
            self.human_total_steps[h.label] += 1
            util = getattr(h, "utilization", 0)
            if isinstance(util, (int, float)) and util >= self.human_overload_thresh:
                self.human_overload_steps[h.label] += 1
            if assignments.get(h) is None:
                self.human_idle_steps[h.label] += 1

        # allocations string
        alloc_str = ";".join(f"{a.label}->{ap}" for a, ap in assignments.items())

        # per-human util + SA snapshot (raw score; add 1.0 on completion; decay each tick)
        human_cols = {}
        for h in agents_by_type.get("humans", []):
            util = getattr(h, "utilization", 0)
            score = max(0.0, float(self.human_SA[h.label]))  # clamp at 0 just in case
            human_cols[f"{h.label}_util"] = util
            human_cols[f"{h.label}_SA"] = score

        row = {
            "t": float(t),
            "unlocked_cnt": int(len(unlocked)),
            "assigned_cnt": int(len(assignments)),
            "completed_cnt": int(len(completed)),
            "allocations": alloc_str,
        }
        row.update(human_cols)
        self.ts_rows.append(row)
        self.t_last = t

    def set_mission_completed(self, t):
        if self.mission_completed_t is None:
            self.mission_completed_t = t

    def dump(self, tag="episode"):
        ts = strftime("%Y%m%d_%H%M%S")
        stem = f"{tag}_{ts}"
        unified_path = self.out_dir / f"{stem}_unified.csv"

        unified_rows = []

        # 1) step rows
        for r in self.ts_rows:
            rr = {"table": "step"}; rr.update(r)
            unified_rows.append(rr)

        # 2) ap_event rows (latencies)
        all_aps = set(self.ap_unlock_t) | set(self.ap_first_assign_t) | set(self.ap_complete_t)
        for ap in sorted(all_aps):
            tu = self.ap_unlock_t.get(ap)
            ta = self.ap_first_assign_t.get(ap)
            tc = self.ap_complete_t.get(ap)
            unified_rows.append({
                "table": "ap_event",
                "ap": ap,
                "t_unlock": tu,
                "t_first_assign": ta,
                "t_complete": tc,
                "lat_unlock_to_assign": None if (tu is None or ta is None) else (ta - tu),
                "lat_assign_to_complete": None if (ta is None or tc is None) else (tc - ta),
                "first_assignee": self.ap_assigned_to.get(ap),
            })

        # 3) agent rows (distance + switches + overload/idle for humans)
        agent_labels = set(self.agent_dist) | set(self.agent_switches) | set(self.human_total_steps)
        for lbl in sorted(agent_labels):
            # overload/idle only meaningful for human labels we tracked
            total = self.human_total_steps.get(lbl, 0)
            overload_ratio = (self.human_overload_steps[lbl] / total) if total > 0 else None
            idle_ratio = (self.human_idle_steps[lbl] / total) if total > 0 else None

            unified_rows.append({
                "table": "agent",
                "agent": lbl,
                "distance": float(self.agent_dist.get(lbl, 0.0)),
                "switches": int(self.agent_switches.get(lbl, 0)),
                "overload_ratio": overload_ratio,
                "idle_ratio": idle_ratio,
            })

        # 4) episode summary row
        unified_rows.append({
            "table": "episode",
            "mission_completion_time": self.mission_completed_t,
            "episode_duration": self.t_last - (self.t0 or 0.0),
            "num_APs_unlocked": len(self.ap_unlock_t),
            "num_APs_assigned": len(self.ap_first_assign_t),
            "num_APs_completed": len(self.ap_complete_t),
        })

        # write single CSV
        cols = sorted({k for r in unified_rows for k in r.keys()}, key=lambda k: (k != "table", k))
        with unified_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in unified_rows:
                w.writerow(r)

        # (Optional) also write separate files for debugging/backward-compat
        if self.also_write_separate:
            base = self.out_dir / stem
            # time-series
            if self.ts_rows:
                cols_ts = sorted({k for r in self.ts_rows for k in r.keys()}, key=lambda k: (k != "t", k))
                with (base.with_name(base.name + "_timeseries.csv")).open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=cols_ts); w.writeheader(); w.writerows(self.ts_rows)
            # ap events
            ap_rows = [r for r in unified_rows if r.get("table") == "ap_event"]
            if ap_rows:
                with (base.with_name(base.name + "_ap_events.csv")).open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(ap_rows[0].keys())); w.writeheader(); w.writerows(ap_rows)
            # summary json
            summary = {
                "mission_completion_time": self.mission_completed_t,
                "episode_duration": self.t_last - (self.t0 or 0.0),
                "agent_distance": dict(self.agent_dist),
                "agent_switches": dict(self.agent_switches),
                "num_APs": {
                    "unlocked": len(self.ap_unlock_t),
                    "assigned": len(self.ap_first_assign_t),
                    "completed": len(self.ap_complete_t),
                },
            }
            with (base.with_name(base.name + "_summary.json")).open("w") as f:
                json.dump(summary, f, indent=2)

        print(f"[metrics] wrote {unified_path}")
        return {"unified_csv": str(unified_path)}
    

def run_one_episode(seed=None,
                    log_dir=LOG_DIR,
                    sliding_window=SLIDING_WINDOW,
                    time_out=TIME_OUT,
                    grid=grid_size,
                    fire_times=FIREMSG_TIMES,
                    surv_times=SURVIVORMSG_TIMES,
                    atm_times=ATMMSG_TIMES,
                    show_episode_plots=True,
                    animate_episode=True,
                    enable_manual_pick=False):
    """
    Runs one episode and returns a dict with {'unified_csv': <path>}.

    Behavior matches your current main when:
      - show_episode_plots=True
      - animate_episode=True
      - enable_manual_pick=False
    """
    # --- Seeding (optional for reproducibility across episodes) ---------------
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))

    # Event variables
    firemsg_idx = 0
    survivormsg_idx = 0
    atmmsg_idx = 0

    # Survivor search
    survivor_scanned = set()

    # Search regions
    s_mask = [1] * 15

    # Setup binding manager
    binding_mgr = BindingManager(verbose=False)

    # Setup specification
    spec = Specification()
    spec.get_task_specification("Case2", s=s_mask, binding_manager=binding_mgr)

    # Setup workspace
    ws = Workspace(size=grid, target_mask=s_mask, num_drones=3, num_gvs=4, num_humans=2, margin=4)

    # Agents
    agents_by_type = {
        "drones": ws.agents["drones"],
        "gvs": ws.agents["gvs"],
        "humans": ws.agents["humans"]
    }
    binding_mgr.agents_by_type = agents_by_type

    # Setup labeler
    labeler = Labeler(spec)

    # Setup allocator (placeholder)
    allocator = RandomAllocator(spec, agents_by_type, binding_mgr, labeler, ws)

    # Simulation
    sim = Simulation(spec, ws, allocator, labeler)

    # Target locations for re-allocation (GUI-like mirror)
    tasks = [[i+1, ws.target_locations[i], ws.get_target_priority(i)] for i in range(len(ws.target_locations))]
    firemsg_idx = 0

    # Metric recorder
    rec = MetricsRecorder(out_dir=log_dir, window=sliding_window)
    rec.start_episode(t0=0.0, agents=ws.get_all_agents())

    # Human agents: new fields
    for human in ws.agents["humans"]:
        human.util_history = []
        human.last_state = None

    # Previous assignment
    prev_assignments = []

    # Flags
    running = True
    verbose = False
    last_print_time = 0

    # Animation traces
    for a in ws.get_all_agents():
        a.traj = []
        a.progress_traj = []
        a.current_symbolic_task_traj = []

    # Initialize time
    prev_time = 0.0
    init_time = 0.0
    running_time = 0.0
    dt = 2.0    # Discrete time (main)

    # ------------------------------- Main loop --------------------------------
    while running:
        # Compute time
        current_time = prev_time + dt
        prev_time += dt
        running_time = current_time - init_time

        # Step the simulation
        sim_outputs = sim.step(dt=dt, mode="sim", verbose=False)
        unlocked   = sim_outputs["unlocked"]
        assignments= sim_outputs["assignments"]
        completed  = sim_outputs["completed"] 

        # Metrics: per-step updates
        rec.note_unlocks(unlocked, running_time)
        rec.note_assignments(assignments, running_time)
        rec.note_completions(completed, running_time, agents_by_type)

        # per-agent distance
        rec.update_distances(ws.get_all_agents())

        # SA decay then record step
        rec.decay_SA(agents_by_type)
        rec.record_step(
            t=running_time,
            assignments=assignments,
            unlocked=unlocked,
            completed=completed,
            agents_by_type=agents_by_type
        )

        # Animation traces
        for a in ws.get_all_agents():
            a.traj.append(tuple(a.pos))  # record position each step
            task = getattr(a, "current_symbolic_task", None)
            prog = float(a.get_progress(task)) if task else 0.0
            a.progress_traj.append(prog)
            a.current_symbolic_task_traj.append(task)

        # Print assignments
        if assignments != prev_assignments:
            print(f"Assigned: {[f'{a.label}→{ap}' for a, ap in assignments.items()]}")
            prev_assignments = assignments
        
        # Human assignment history & utilization
        for human in ws.agents["humans"]:

            # Seed once
            if not getattr(human, "util_history", None):
                human.util_history = [(0.0, 'idle')]
                human.last_state = 'idle'

            # 1) Busy vs idle (busy if they have an assignment)
            assigned = assignments.get(human)
            new_state = 'busy' if assigned is not None else 'idle'

            # 2) On state-change, record the timestamp
            if new_state != human.last_state:
                human.util_history.append((running_time, new_state))
                human.last_state = new_state

            # 3) Prune old events, but keep the last before t0
            t0 = running_time - sliding_window
            ev = human.util_history
            older = [e for e in ev if e[0] < t0]
            newer = [e for e in ev if e[0] >= t0]
            keep = ([max(older, key=lambda x: x[0])] if older else []) + newer
            human.util_history = keep

            # 4) Compute % utilization over the last window
            human.utilization = compute_utilization(human, running_time, sliding_window)

            # --- Event triggers -------------------------------------------------
            # 1. FIRE → set priority
            if (firemsg_idx < len(fire_times)) and (running_time >= fire_times[firemsg_idx]):
                all_agents = ws.agents["drones"] + ws.agents["gvs"]
                if all_agents:
                    a = random.choice(all_agents)
                    dists = [np.linalg.norm(np.array(loc) - a.pos[:2]) for loc in ws.target_locations]
                    candidates = np.argsort(dists)[:max(1, len(dists)//3)]
                else:
                    candidates = range(len(ws.target_locations))

                choices = [tid for tid in candidates if ws.get_target_priority(tid) < 2]
                if not choices:
                    choices = list(range(len(ws.target_locations)))
                tid = int(random.choice(choices))
                required = int(np.random.choice([2, 2, 1]))
                ws.set_target_priority(tid, required)
                labeler.advance({"p_firemsg_0_0_0_0"})
                for row in tasks:
                    if row[0] == tid + 1:
                        row[2] = required
                        break
                firemsg_idx += 1

            # 2. Survivor messages
            if (survivormsg_idx < len(surv_times)) and (running_time >= surv_times[survivormsg_idx]):
                labeler.advance({"p_survivormsg_0_0_0_0"})
                survivormsg_idx += 1

            # 3. ATM scheduler
            if (atmmsg_idx < len(atm_times)) and (running_time >= atm_times[atmmsg_idx]):
                labeler.advance({"p_atmmsg_0_0_0_0"})
                atmmsg_idx += 1

            # Survivor scan -> choose gate
            for ap in completed:
                if ap.startswith("p_verify") and ap not in survivor_scanned:
                    target_id = ap.split("_")[2]
                    if random.random() < 0.8:
                        labeler.chosen_gate_per_group[target_id] = f"p_foundgate_{target_id}"
                    else:
                        labeler.chosen_gate_per_group[target_id] = f"p_notfoundgate_{target_id}"
                    chosen_gate = labeler.chosen_gate_per_group.get(target_id)
                    labeler.advance({chosen_gate})

        # Periodic debug
        if verbose and running_time - last_print_time >= 20.0:
            last_print_time = running_time
            print(f"\n[DEBUG:{running_time:.2f}] -------------------------")
            print(f"[DEBUG:{running_time:.2f}] Unlocked APs: {sorted(unlocked)}")
            print(f"[DEBUG:{running_time:.2f}] Assigned: {[f'{a.label}→{ap}' for a, ap in assignments.items()]}")
            print(f"[DEBUG:{running_time:.2f}] Completed: {sorted(completed)}")

        # Check termination
        if labeler.all_completed() and ws.all_mobile_agents_at_base():
            print(f"[t={running_time:.2f}] Mission completed!")
            rec.set_mission_completed(running_time)
            running = False

        # Time out
        if running_time > time_out:
            print(f"[t={running_time:.2f}] Time out!")
            running = False
    
    print("Simulation Completed.")

    # Metrics: dump
    paths = rec.dump(tag="hat_episode")
    print("Saved metrics:", paths)

    # --- Per-episode figures (SA/Util with MCT) -------------------------------
    if show_episode_plots:
        try:
            from ltl_core.visualization import plot_episode_metrics, find_latest_metrics_csv
            latest_csv = paths.get("unified_csv") if isinstance(paths, dict) else None
            if not latest_csv:
                latest_csv = find_latest_metrics_csv(log_dir)
            human_labels = [h.label for h in ws.agents["humans"]]
            plot_episode_metrics(csv_path=latest_csv, human_labels=human_labels, show=False)
            print(f"[viz] Plotted metrics from {latest_csv}")
        except Exception as e:
            print(f"[viz] metrics plotting failed: {e}")

    # --- Animation (replay) ----------------------------------------------------
    if animate_episode:
        max_len = max(len(a.traj) for a in ws.get_all_agents())
        ani = animate_workspace(ws, sim=None, steps=max_len, interval=100, record=False)
        import matplotlib.pyplot as plt  # ensure plt is in scope
        plt.show()
        print("Animation Closed.")

    # --- Optional manual CSV picker -------------------------------------------
    if enable_manual_pick:
        try:
            from ltl_core.visualization import choose_metrics_csv, plot_episode_metrics
            start_dir = str(log_dir)
            picked_csv = choose_metrics_csv(start_dir=start_dir)
            if picked_csv:
                human_labels = [h.label for h in ws.agents["humans"]]
                plot_episode_metrics(csv_path=picked_csv, human_labels=human_labels, show=False)
                import matplotlib.pyplot as plt
                plt.show()
                print(f"[viz] Plotted metrics from manually selected file: {picked_csv}")
            else:
                print("[viz] Manual CSV selection canceled.")
        except Exception as e:
            print(f"[viz] manual CSV picking failed: {e}")

    return paths



if __name__ == '__main__':
    # ===================== Batch controller ===================================
    # Exactly the behavior you asked:
    #  - BATCH_EPISODES == 1  -> behave exactly like current single-episode run
    #  - BATCH_EPISODES >= 2  -> run multiple episodes and show box plots
    #  - otherwise            -> fall back to single-episode run
    BATCH_EPISODES  = 1       # <--- set this as you like
    SAVE_BOX_PLOTS  = False   # save multi-episode figures to LOG_DIR/figs_multi
    PLOT_LATENCIES  = False   # also plot AP latency box plots across episodes

    if BATCH_EPISODES == 1:
        # Measure time
        cpu0, wall0 = time.process_time(), time.perf_counter()
        # Single episode: keep identical behavior
        run_one_episode(
            seed=None,
            log_dir=LOG_DIR,
            show_episode_plots=True,
            animate_episode=True,
            enable_manual_pick=False  # set True if you want the picker at the end
        )
        # Time measurement out
        cpu1, wall1 = time.process_time(), time.perf_counter()
        print(f"Single episode run time: CPU={cpu1-cpu0:.2f}s, Wall={wall1-wall0:.2f}s")

    elif BATCH_EPISODES >= 2:
        # Multi-episode batch
        from pathlib import Path
        from ltl_core.visualization import (
            plot_boxplots_across_episodes,
            plot_ap_latency_boxplots,
        )

        all_csvs = []
        for k in range(BATCH_EPISODES):
            print(f"\n===== Running episode {k+1}/{BATCH_EPISODES} =====")
            paths = run_one_episode(
                seed=1000 + k,           # vary seeds per episode
                log_dir=LOG_DIR,
                show_episode_plots=False,  # suppress per-episode figures
                animate_episode=False,     # no animation in batch
                enable_manual_pick=False
            )
            csv_path = paths.get("unified_csv") if isinstance(paths, dict) else None
            if csv_path:
                all_csvs.append(csv_path)
                print(f"[batch] Collected CSV: {csv_path}")
            else:
                print("[batch] WARNING: No unified CSV for this episode.")

        # After all episodes, draw the box plots you requested
        save_dir = str(Path(LOG_DIR) / "figs_multi") if SAVE_BOX_PLOTS else None
        if all_csvs:
            plot_boxplots_across_episodes(
                csv_paths=all_csvs,
                show=True,
                save_dir=save_dir
            )
            if PLOT_LATENCIES:
                plot_ap_latency_boxplots(
                    csv_paths=all_csvs,
                    show=True,
                    save_dir=save_dir
                )
        else:
            print("[batch] No CSVs collected; nothing to plot.")

    else:
        # Fallback (e.g., BATCH_EPISODES == 2) -> single run
        run_one_episode(
            seed=None,
            log_dir=LOG_DIR,
            show_episode_plots=True,
            animate_episode=True,
            enable_manual_pick=False
        )
