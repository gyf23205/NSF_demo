import sys
sys.path.append('C:/Users/sooyung/Research/NSF_demo')
from ltl_core.binding_manager import *
from ltl_core.specification import *
from ltl_core.workspace import *
from ltl_core.labeler import *
from ltl_core.visualization import *
from ltl_core.allocator import *
from ltl_core.simulation import *


if __name__ == '__main__':
    # Search regions
    s_mask = [1] * 2

    # Setup binding manager
    binding_mgr = BindingManager(verbose=False)

    # Setup specification
    spec = Specification()
    spec.get_task_specification("Case2", s=s_mask, binding_manager=binding_mgr)

    # Setup workspace
    ws = Workspace(size=(50, 40), target_mask=s_mask, num_drones=4, num_gvs=2, num_humans=2, margin=4)

    # Agents
    agents_by_type = {
        "drone": ws.agents["drones"],
        "gv": ws.agents["gvs"],
        "human": ws.agents["humans"]
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

    # Initialize simulation time
    prev_time = 0.0
    init_time = 0.0
    running_time = 0.0
    dt = 0.1    # Discrete time

    # Main loop
    while running:
        # Compute time
        current_time = prev_time + dt
        prev_time += dt
        running_time = current_time - init_time

        # Check if simulation is done
        if labeler.all_completed():
            print(f"[t={running_time:.2f}] Mission completed!")
            running = False

        # Step the simulation
        


