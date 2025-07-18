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
    # === Mission Specification ===
    case = "Case2"
    s_mask = [1, 1, 1, 1, 1]

    # === Setup binding manager and LTL-based Specification ===
    binding_manager = BindingManager(verbose=False)
    spec = Specification()
    spec.get_task_specification(case, s=s_mask, binding_manager=binding_manager)

    # === Initialize GUI ===
    pygame.init()
    # screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("LTL-based GUI")
    clock = pygame.time.Clock()
    running = True

    # === Workspace Setup ===
    ws = Workspace(size=grid_size, target_mask=s_mask, margin=4)
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

    # The main loop for the GUI
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # === Compute timestep ===
        current_time = time()
        dt = current_time - prev_time
        prev_time = current_time

        # === Step the simulation ===
        sim_outputs = sim.step(dt=dt, verbose=False)

        # Optional: extract info for debug or visual overlays
        unlocked = sim_outputs["unlocked"]
        assignments = sim_outputs["assignments"]
        completed = sim_outputs["completed"]

        # === Draw GUI: simple ===
        # draw_workspace(screen, ws, screensize=screen_size)

        # === Draw GUI: using GameMgr ===
        for agent, visual in agent_to_visual.items():
            pos = agent.pos
            pos = ws.grid_to_game_mgr(pos, grid_size=grid_size)
            if hasattr(agent, "role") and agent.role == "drones":
                # Drones need altitude
                if len(pos) == 2:
                    pos = np.append(pos, 0.0)
                pos = np.append(pos, 0.0)
                visual.position = tuple(pos)
            else:
                # GVs and others are fine with 2D
                visual.position = tuple(pos[:2])
        # Render the game manager
        game_mgr.render(vor, centroids)

        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()
