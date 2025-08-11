import sys
sys.path.append('C:/Users/sooyung/Research/NSF_demo')
import random
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
FIREMSG_TIMES = [30.0, 45.0, 80.0, 110.0]
SURVIVORMSG_TIMES = [25.0, 55.0, 70.0, 100.0]
ATMMSG_TIMES = [35.0, 65.0, 90.0, 120.0]


def compute_utilization(human, now, window=SLIDING_WINDOW):
    # start from window‐ago
    t0 = now - window
    busy_time = 0.0
    prev_t, prev_s = t0, 'idle'

    # walk through the history in order
    for t, s in sorted(human.util_history, key=lambda x: x[0]):
        if prev_s == 'busy':
            busy_time += t - prev_t
        prev_t, prev_s = t, s

    # account for final segment up to now
    if prev_s == 'busy':
        busy_time += now - prev_t

    pct = int(100 * busy_time / window)
    return max(0, min(100, pct))


if __name__ == '__main__':
    # Event variables
    firemsg_idx = 0
    survivormsg_idx = 0
    atmmsg_idx = 0

    # Survivor search
    survivor_scanned = set()

    # Search regions
    s_mask = [1] * 10

    # Setup binding manager
    binding_mgr = BindingManager(verbose=False)

    # Setup specification
    spec = Specification()
    spec.get_task_specification("Case2", s=s_mask, binding_manager=binding_mgr)

    # Setup workspace
    ws = Workspace(size=grid_size, target_mask=s_mask, num_drones=4, num_gvs=2, num_humans=2, margin=4)

    # Agents
    agents_by_type = {
        "drones": ws.agents["drones"],
        "gvs": ws.agents["gvs"],
        "humans": ws.agents["humans"]
    }
    binding_mgr.agents_by_type = agents_by_type

    # Setup labeler
    labeler = Labeler(spec)

    # Draw: Hierarchical structure
    # draw_composite_hierarchy(spec)

    # Setup random allocator (place holder)
    allocator = RandomAllocator(spec, agents_by_type, binding_mgr, labeler)

    # Simulation
    sim = Simulation(spec, ws, allocator, labeler)

    # Human agents: new fields
    for human in ws.agents["humans"]:
        human.util_history = []
        human.last_state = None

    # Previous assignment
    prev_assignments = []

    # Flag(s)
    running = True

    # Print
    verbose = False
    last_print_time = 0

    # Animation
    for a in ws.get_all_agents():
        a.traj = []
        a.progress_traj = []
        a.current_symbolic_task_traj = []

    # Initialize simulation time
    prev_time = 0.0
    init_time = 0.0
    running_time = 0.0
    dt = 1.0    # Discrete time (main)

    # Main loop
    while running:
        # Compute time
        current_time = prev_time + dt
        prev_time += dt
        running_time = current_time - init_time

        # Step the simulation
        sim_outputs = sim.step(dt=dt, mode="sim", verbose=False)
        # Atomic propositions
        unlocked = sim_outputs["unlocked"]
        assignments = sim_outputs["assignments"]
        completed = sim_outputs["completed"] 

        # Animation
        for a in ws.get_all_agents():
            a.traj.append(tuple(a.pos))  # record position each step

            # record progress for the current symbolic task
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
            # 1) Detect busy vs idle (busy if they _have_ an assignment)
            assigned = assignments.get(human)
            new_state = 'busy' if assigned is not None else 'idle'

            # 2) On state‐change, record the timestamp
            if new_state != human.last_state:
                human.util_history.append((running_time, new_state))
                human.last_state = new_state

            # 3) Prune old events outside the sliding window
            human.util_history = [
                (t,s) for t,s in human.util_history
                if running_time - t <= SLIDING_WINDOW
            ]

            # 4) Compute % utilization over the last window
            human.utilization = compute_utilization(human, running_time, SLIDING_WINDOW)

            # Monitor APs triggers
            # 1. possible new emergency events (FIRE → ask human to set priority)
            if (firemsg_idx < len(FIREMSG_TIMES)
                    and running_time >= FIREMSG_TIMES[firemsg_idx]):
                labeler.advance({"p_firemsg_0_0_0_0"})
                firemsg_idx += 1

            # 2. possible new survivor messages
            if (survivormsg_idx < len(SURVIVORMSG_TIMES)
                    and running_time >= SURVIVORMSG_TIMES[survivormsg_idx]):
                labeler.advance({"p_survivormsg_0_0_0_0"})
                survivormsg_idx += 1

            # 3. ATM scheduler (prompt + env broadcast)
            if (atmmsg_idx < len(ATMMSG_TIMES)
                     and running_time >= ATMMSG_TIMES[atmmsg_idx]):
                labeler.advance({"p_atmmsg_0_0_0_0"})
                atmmsg_idx += 1

            # Survivor scan -> choose gate
            for ap in completed:
                if ap.startswith("p_verify") and ap not in survivor_scanned:
                    target_id = ap.split("_")[2]

                    # Chose gate
                    if random.random() < 0.8:
                        labeler.chosen_gate_per_group[target_id] = f"p_foundgate_{target_id}"
                    else:
                        labeler.chosen_gate_per_group[target_id] = f"p_notfoundgate_{target_id}"

                    # Advance
                    chosen_gate = labeler.chosen_gate_per_group.get(target_id)
                    labeler.advance({chosen_gate})

        # Print
        if verbose and running_time - last_print_time >= 20.0:
            last_print_time = running_time
            print(f"\n[DEBUG:{running_time:.2f}] -------------------------")
            print(f"[DEBUG:{running_time:.2f}] Unlocked APs: {sorted(unlocked)}")
            print(f"[DEBUG:{running_time:.2f}] Assigned: {[f'{a.label}→{ap}' for a, ap in assignments.items()]}")
            print(f"[DEBUG:{running_time:.2f}] Completed: {sorted(completed)}")

        # Check if simulation is done
        if labeler.all_completed() and ws.all_mobile_agents_at_base():
            print(f"[t={running_time:.2f}] Mission completed!")
            running = False

        # Time out
        if running_time > TIME_OUT:
            print(f"[t={running_time:.2f}] Time out!")
            running = False
    
    print("Simulation Completed.")

    # Episode finished – replay from the recorded traces
    max_len = max(len(a.traj) for a in ws.get_all_agents())
    ani = animate_workspace(ws, sim=None, steps=max_len, interval=100, record=False)
    plt.show()

    print("Animation Closed.")
