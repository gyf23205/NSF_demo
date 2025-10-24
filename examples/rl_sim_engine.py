import sys
from pathlib import Path
my_path = str(Path(__file__).resolve().parent)
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

class MetricsRecorder:
    """
    Record metrics during the simulation as a CSV file.
    """
    def __init__(self, out_dir, window=60.0, human_overload_thresh=0.8, also_write_separate=False):
        self.out_dir = Path(out_dir).expanduser()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.also_write_separate = also_write_separate

        # Time-series
        self.ts_rows = []

        # AP lifecycle
        self.ap_unlock_t = {}           # ap -> time unlocked
        self.ap_first_assign_t = {}     # ap -> time first assigned
        self.ap_complete_t = {}         # ap -> time completed
        self.ap_assigned_to = {}        # ap -> agent label

        # Per-agent stats
        self.agent_switches = defaultdict(int)  # Symbolic-task switches
        self.agent_prev_task = {}

        # Per-human SA proxy (TO DO: improve this)
        self.human_SA = defaultdict(float)
        self.SA_decay = 0.98
        self.SA_credit = 1.0

        # Per-agent distance
        self.agent_prev_pos = {}
        self.agent_dist = defaultdict(float)

        # Human overload / idle tracking
        self.human_overload_thresh = human_overload_thresh  # utilization is in percent (0..100)
        self.human_overload_steps  = defaultdict(int)       # label -> count
        self.human_idle_steps      = defaultdict(int)       # label -> count
        self.human_total_steps     = defaultdict(int)       # label -> count

        # Timing
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
        # assignments: {agent_ojb: ap}
        for agent, ap in assignments.items():
            if ap not in self.ap_first_assign_t:
                self.ap_first_assign_t[ap] = t
                self.ap_assigned_to[ap] = agent.label
            # Count symbolic task switches
            cur = getattr(agent, "current_symbolic_task", None)
            prev = self.agent_prev_task.get(agent.label, None)
            if cur is not None and prev is not None and cur != prev:
                self.agent_switches[agent.label] += 1
            self.agent_prev_task[agent.label] = cur

    def note_completions(self, completed, t, agents_by_type):
        completed_set = set(completed)

        # Credit +1.0 to the recorded human assignee for each completed AP
        human_labels = {h.label for h in agents_by_type.get("humans", [])}
        for ap in completed_set:
            assignee = self.ap_assigned_to.get(ap, None)
            if assignee in human_labels:
                self.human_SA[assignee] += 1.0

        # Mark completion time for first time only
        for ap in completed_set:
            if ap not in self.ap_complete_t:
                self.ap_complete_t[ap] = t

    def decay_SA(self, agents_by_type):
        for h in agents_by_type.get("humans", []):
            self.human_SA[h.label] *= self.SA_decay

    def update_distances(self, agents):
        for a in agents:
            pl = self.agent_prev_pos.get(a.label)
            if pl is not None:
                dx = float(a.pos[0] - pl[0]); dy = float(a.pos[1] - pl[1])
                self.agent_dist[a.label] += math.hypot(dx, dy)
            self.agent_prev_pos[a.label] = tuple(a.pos)

    def set_mission_completed(self, t):
        if self.mission_completed_t is None:
            self.mission_completed_t = t

    def record_step(self, t, assignments, unlocked, completed, agents_by_type):
        # Per-human overload/idle counters
        for h in agents_by_type.get("humans", []):
            self.human_total_steps[h.label] += 1
            util = getattr(h, "utilization", 0)
            if isinstance(util, (int, float)) and util >= self.human_overload_thresh:
                self.human_overload_steps[h.label] += 1
            if assignments.get(h) is None:
                self.human_idle_steps[h.label] += 1

        # Allocations string
        alloc_str = ";".join(f"{a.label}->{ap}" for a, ap in assignments.items())

        # Per-human util + SA snapshot (raw score; add 1.0 on completion; decay each tick)
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


class EventScheduler:
    """Handles stochastic timing and execution of message events (fire/survivor/ATM)."""
    def __init__(self, ws, labeler, T_end, 
                 rate_fire=1/60, rate_surv=1/70, rate_atm=1/80,
                 spatial_temp=0.5, seed=None):
        """
        Args:
            ws, labeler: workspace and labeler objects
            T_end (float): simulation duration (s)
            rate_*: Poisson rate parameters (events per second)
            spatial_temp (float): spatial softening (0=uniform, low=strongly local)
            seed (int): RNG seed
        """
        self.ws = ws
        self.labeler = labeler
        self.rng = np.random.default_rng(seed)
        self.T_end = T_end
        self.spatial_temp = spatial_temp

        # Pre-generate event times
        self.fire_times = self._poisson(rate_fire)
        self.surv_times = self._poisson(rate_surv)
        self.atm_times  = self._poisson(rate_atm)
        self.fire_idx = self.surv_idx = self.atm_idx = 0
        self.survivor_scanned = set()

    # ------------------------------------------------------------------ #
    def _poisson(self, rate_lambda):
        t, times = 0.0, []
        while True:
            t += self.rng.exponential(1/rate_lambda)
            if t > self.T_end: break
            times.append(t)
        return times

    # ------------------------------------------------------------------ #
    def _weighted_target_choice(self, agents):
        """Choose a target near active agents with soft distance weighting."""
        locs = np.array(self.ws.target_locations)
        if not agents or len(locs) == 0:
            return int(self.rng.integers(len(locs)))

        # Compute average position of all mobile agents
        pos = np.array([a.pos[:2] for a in agents])
        center = pos.mean(axis=0)

        # Distance-based soft weighting
        dists = np.linalg.norm(locs - center, axis=1)
        weights = np.exp(-self.spatial_temp * dists)
        weights /= weights.sum()
        return int(self.rng.choice(np.arange(len(locs)), p=weights))

    # ------------------------------------------------------------------ #
    def step(self, t, completed):
        """Check all event streams at time t and trigger those whose time has arrived."""
        if self.fire_idx < len(self.fire_times) and t >= self.fire_times[self.fire_idx]:
            self._trigger_fire()
            self.fire_idx += 1

        if self.surv_idx < len(self.surv_times) and t >= self.surv_times[self.surv_idx]:
            self.labeler.advance({"p_survivormsg_0_0_0_0"})
            self.surv_idx += 1

        if self.atm_idx < len(self.atm_times) and t >= self.atm_times[self.atm_idx]:
            self.labeler.advance({"p_atmmsg_0_0_0_0"})
            self.atm_idx += 1

        self._trigger_verify_gates(completed)

    # ------------------------------------------------------------------ #
    def _trigger_fire(self):
        """Spawn a new fire message: select location, set priority, advance automaton."""
        ws, labeler = self.ws, self.labeler
        field_agents = ws.agents.get("drones", []) + ws.agents.get("gvs", [])
        tid = self._weighted_target_choice(field_agents)
        required = int(self.rng.choice([2, 2, 1]))  # mostly priority 2
        ws.set_target_priority(tid, required)
        labeler.advance({"p_firemsg_0_0_0_0"})

    # ------------------------------------------------------------------ #
    def _trigger_verify_gates(self, completed):
        """After verification APs complete, choose found/notfound gates."""
        for ap in completed:
            if ap.startswith("p_verify") and ap not in self.survivor_scanned:
                target_id = ap.split("_")[2]
                if random.random() < 0.8:
                    gate = f"p_foundgate_{target_id}"
                else:
                    gate = f"p_notfoundgate_{target_id}"
                self.labeler.chosen_gate_per_group[target_id] = gate
                self.labeler.advance({gate})
                self.survivor_scanned.add(ap)


def run_one_episode(seed=None,
                    log_dir=LOG_DIR,
                    slide_window=SLIDING_WINDOW,
                    time_out=TIME_OUT,
                    grid=grid_size,
                    n_region=15,
                    show_episode_plots=True,
                    animate_episode=True,
                    enable_manual_pick=False):
    """
    Run one episode of the simulation with given parameters.
    Return a dict with {'unified_csv': <path>}.
    """

    # Seeding
    if seed is not None:
        np.random.seed(int(seed))
        random.seed(int(seed))

    # Search regions
    s_mask = [1] * n_region

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

    # Event Scheduler
    scheduler = EventScheduler(ws, labeler, T_end=TIME_OUT, rate_fire=1/60, rate_surv=1/70, rate_atm=1/90)

    # Setup allocator (placeholder)
    allocator = RandomAllocator(spec, agents_by_type, binding_mgr, labeler, ws)

    # Simulation
    sim = Simulation(spec, ws, allocator, labeler)

    # Metric recorder
    rec = MetricsRecorder(out_dir=log_dir, window=slide_window)
    rec.start_episode(t0=0.0, agents=ws.get_all_agents())

    # Human agents
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

    # Initialize time and discrete time
    prev_time = 0.0
    init_time = 0.0
    running_time = 0.0
    dt = 1.0

    # ------------------------------- Main loop --------------------------------
    while running:
        # Compute time
        current_time = prev_time + dt
        prev_time += dt
        running_time = current_time - init_time

        # Step the simulation
        sim_outputs = sim.step(dt=dt, mode="sim", verbose=verbose)
        unlocked    = sim_outputs["unlocked"]
        assignments = sim_outputs["assignments"]
        completed   = sim_outputs["completed"] 

        # Metric recording
        rec.note_unlocks(unlocked, running_time)
        rec.note_assignments(assignments, running_time)
        rec.note_completions(completed, running_time, agents_by_type)
        rec.update_distances(ws.get_all_agents())
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
            a.traj.append(tuple(a.pos))
            task = getattr(a, "current_symbolic_task", None)
            prog = float(a.get_progress(task)) if task else 0.0
            a.progress_traj.append(prog)
            a.current_symbolic_task_traj.append(task)

        # Print assignments and update previous assignments
        if assignments != prev_assignments:
            print(f"[Time {current_time:.1f}s] Assigned: {[f'{a.label}->{ap}' for a, ap in assignments.items()]}")
            prev_assignments = assignments

        # Human assignment history and utilization
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
            t0 = running_time - slide_window
            ev = human.util_history
            older = [e for e in ev if e[0] < t0]
            newer = [e for e in ev if e[0] >= t0]
            keep = ([max(older, key=lambda x: x[0])] if older else []) + newer
            human.util_history = keep

            # 4) Compute utilization over the last window
            human.utilization = compute_utilization(human, running_time, slide_window)

        # Step event scheduler
        scheduler.step(running_time, completed)

        # Periodic debug
        if verbose and running_time - last_print_time >= 20.0:
            last_print_time = running_time
            print(f"\n[DEBUG:{running_time:.2f}] -------------------------")
            print(f"[DEBUG:{running_time:.2f}] Unlocked APs: {sorted(unlocked)}")
            print(f"[DEBUG:{running_time:.2f}] Assigned: {[f'{a.label}->{ap}' for a, ap in assignments.items()]}")
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

    # --- Per-episode figures --------------------------------------------------
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
        ani = animate_workspace(ws, sim=None, steps=max_len, interval=100, record=True)
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
                plt.show()
                print(f"[viz] Plotted metrics from manually selected file: {picked_csv}")
            else:
                print("[viz] Manual CSV selection canceled.")
        except Exception as e:
            print(f"[viz] manual CSV picking failed: {e}")

    return paths


if __name__ == "__main__":
    run_one_episode(
        seed=None,
        log_dir=LOG_DIR,
        n_region=15,
        show_episode_plots=True,
        animate_episode=True,
        enable_manual_pick=False
    )
