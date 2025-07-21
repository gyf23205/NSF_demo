import sys
sys.path.append('C:/Users/sooyung/Research/NSF_demo')
import pygame
from time import time, sleep, strftime
from gui_panel_ltl import GameMgr
from vehicles import VirtualDrone, VirtualGV
from scipy.spatial import Voronoi
import numpy as np
from ltl_core.specification import Specification
from ltl_core.binding_manager import BindingManager
from ltl_core.workspace import Workspace
from ltl_core.dag_builder import build_dag
from ltl_core.automaton_generator import compile_automata
from ltl_core.labeler import Labeler
from ltl_core.allocator import RandomAllocator
from ltl_core.simulation import Simulation
from ltl_core.visualization import draw_workspace

grid_size = (50, 40)
# screen_size = (grid_size[0] * 30, grid_size[1] * 30)

if __name__ == "__main__":
    # === Initialize GUI ===
    pygame.init()
    # screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("LTL-based GUI")
    clock = pygame.time.Clock()
    running = True
    landing = False

    # === Constants ===
    hover_duration = 10
    n_targets = 10
    n_drones = 4
    n_gvs = 2
    n_humans = 2

    # === Mission Specification ===
    case = "Case2"
    s_mask = [1] * n_targets

    # === Setup binding manager and LTL-based Specification ===
    binding_manager = BindingManager(verbose=False)
    spec = Specification()
    spec.get_task_specification(case, s=s_mask, binding_manager=binding_manager)

    # === Workspace Setup ===
    ws = Workspace(size=grid_size, target_mask=s_mask, num_drones=n_drones, num_gvs=n_gvs, num_humans=n_humans, margin=4)
    centroids = ws.target_locations
    vor = Voronoi(np.array(centroids))

    # === Agents and Binding Manager ===
    agents_by_type = {
        "drone": ws.agents["drones"],
        "gv": ws.agents["gvs"],
        "human": ws.agents["humans"]
    }
    binding_manager.agents_by_type = agents_by_type

    # === Create visual agents from symbolic agents ===
    drones = []
    gvs = []
    agent_to_visual = {}

    for i, agent in enumerate(agents_by_type["drone"]):
        pos = agent.pos
        if len(pos) == 2:
            pos = np.append(pos, 0.0)  # Add dummy altitude
        vd = VirtualDrone(i, tuple(pos))
        drones.append(vd)
        agent_to_visual[agent] = vd
    for i, agent in enumerate(agents_by_type["gv"]):
        pos = agent.pos
        gv = VirtualGV(i, tuple(pos))
        gvs.append(gv)
        agent_to_visual[agent] = gv

    # === GameMgr for rendering ===
    game_mgr = GameMgr(drones, gvs, ws)

    # === Get drone start positions in GUI space ===
    takeoff_positions = [agent.pos[:2] for agent in agents_by_type["drone"]]
    takeoff_gui = [ws.grid_to_pixel(pos, grid_size=(50, 40), screen_size=(900, 720)) for pos in takeoff_positions]
    game_mgr.set_takeoff_positions(takeoff_gui)

    # === Task interpretation ===
    tasks = []
    for idx, pos in enumerate(ws.target_locations):
        gui_pos = ws.grid_to_pixel(pos, grid_size=(50, 40), screen_size=(900, 720))
        tasks.append([idx + 1, list(gui_pos), 0, 0])
    game_mgr.set_task(tasks)

    # === Labeler setup ===
    labeler = Labeler(spec, spec.dag, spec.automata)

    # === Allocator ===
    allocator = RandomAllocator(spec, agents_by_type, binding_manager, labeler)

    # === Simulation ===
    sim = Simulation(spec, ws, allocator, labeler)

    # ==- Initialize simulation time ===
    prev_time = time()
    init_time = time()

    # The main loop for the GUI
    while running:
        # === Avoid high CPU usage ===
        sleep(0.01)

        # === Compute timestep ===
        current_time = time()
        dt = current_time - prev_time
        prev_time = current_time
        running_time = current_time - init_time

        # === Event handling ===
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print(f"[t={running_time:02f}] Quit by window close.")
                running = False
                landing = False

        # === Key handling ===
        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            print(f"[t={running_time:02f}] Escape by keyboard.")
            running = False
            landing = True

        # === Check if simulation is done ===
        if labeler.all_completed():
            print(f"[t={running_time:02f}] Mission completed!")
            running = False
            landing = True

        # === Step the simulation ===
        if running_time < hover_duration:
            for drone in drones:
                drone.takeoff_in_place(1.4)
        else:
            sim_outputs = sim.step(dt=dt, verbose=False)
            # Optional: atomic propositions
            unlocked = sim_outputs["unlocked"]
            assignments = sim_outputs["assignments"]
            completed = sim_outputs["completed"]

        # === Drone/GV positions ===
        for agent, visual in agent_to_visual.items():
            pos = agent.pos
            pos = ws.grid_to_game_mgr(pos, grid_size=grid_size)
            if hasattr(agent, "role") and agent.role == "drones":
                # Drones need altitude
                visual.position = np.append(pos, visual.position[2])
            else:
                # GVs and others are fine with 2D
                visual.position = pos[:2]
            
        # === Awareness map ===
        pos_aware = [(d.position[0], d.position[1]) for d in drones]
        pos_aware = game_mgr.position_meter_to_gui(pos_aware)
        game_mgr.update_awareness(pos_aware, radius=40)

        # === Draw GUI: simple ===
        # draw_workspace(screen, ws, screensize=screen_size)

        # === Draw GUI: Render the game manager ===
        game_mgr.render(vor, centroids)

    # === Drone landing ===
    if landing:
        print(f"[t={running_time:02f}] Landing drones...")
        # Wait until all drones are landed
        while max([drones[idx].position[2] for idx in range(n_drones)]) > 0.01:
            for idx in range(n_drones):
                drones[idx].land_in_place()
                drones[idx].status = 'landing'
                drones[idx].rt = np.random.normal(0, 0.01, 1)[0]
                sleep(0.01)
            game_mgr.render(vor, centroids)
    
    pygame.quit()
