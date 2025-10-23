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

# Event setup [TO DO: ranodmize these]
FIREMSG_TIMES = [30.0, 45.0, 80.0, 110.0, 150.0, 250.0, 300.0]
SURVIVORMSG_TIMES = [25.0, 55.0, 70.0, 100.0, 156.0, 169.3, 264.5]
ATMMSG_TIMES = [35.0, 65.0, 90.0, 120.0, 154.2, 185.3, 210.4, 290.0]


# class MetricsRecorder:


def run_one_episode(seed=None,
                    log_dir=LOG_DIR,
                    slide_window=SLIDING_WINDOW,
                    time_out=TIME_OUT,
                    grid=grid_size,
                    n_region=15,
                    fire_times=FIREMSG_TIMES,
                    surv_times=SURVIVORMSG_TIMES,
                    atm_times=ATMMSG_TIMES,
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

    # Event variables
    firemsg_idx = 0
    survivormsg_idx = 0
    atmmsg_idx = 0

    # Survivor search
    survivor_scanned = set()

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

    # Setup allocator (placeholder)
    allocator = RandomAllocator(spec, agents_by_type, binding_mgr, labeler, ws)

    # Simulation
    sim = Simulation(spec, ws, allocator, labeler)

    # Regions for GUI and priority
    tasks = [[i+1, ws.target_locations[i], ws.get_target_priority(i)] for i in range(len(ws.target_locations))]

    # Metric recorder [TO DO: activate this]
    # rec = MetricsRecorder(out_dir=log_dir, window=sliding_window)
    # rec.start_episode(t0=0.0, agents=ws.get_all_agents())

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

        # Metric recording [TO DO: activate this]

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

        # --- Event triggers -------------------------------------------------
        # 1. Fire -> set priority (only for field agents)
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

        # Survivor scan -> choose gate (80% found, 20% not found)
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
            print(f"[DEBUG:{running_time:.2f}] Assigned: {[f'{a.label}->{ap}' for a, ap in assignments.items()]}")
            print(f"[DEBUG:{running_time:.2f}] Completed: {sorted(completed)}")

        # Check termination
        if labeler.all_completed() and ws.all_mobile_agents_at_base():
            print(f"[t={running_time:.2f}] Mission completed!")
            # [TO DO: Activate this]
            # rec.set_mission_completed(running_time)
            running = False

        # Time out
        if running_time > time_out:
            print(f"[t={running_time:.2f}] Time out!")
            running = False
    
    print("Simulation Completed.")

    # [TO DO: From line 496 to the end of the previous code]



if __name__ == "__main__":
    run_one_episode(
        seed=None,
        log_dir=LOG_DIR,
        show_episode_plots=True,
        animate_episode=True,
        enable_manual_pick=False
    )
