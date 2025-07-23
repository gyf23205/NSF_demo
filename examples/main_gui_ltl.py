import sys
sys.path.append('C:/Users/sooyung/Research/NSF_demo')
import pygame
from time import time, sleep, strftime
from gui_panel_ltl import GameMgr
from vehicles import VirtualDrone, VirtualGV
from scipy.spatial import Voronoi
import numpy as np
import socket
import json
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


def update_wind(game_mgr, wind_time, old_avg_speed, n_wind, message, threshold=1.0):
    """
    Update the wind field in the environment and return the updated values.
    """
    import numpy as np
    from time import time

    new_time = time()

    if not game_mgr.wind:
        wind = [[
            np.random.uniform(-1.9, 1.9),
            np.random.uniform(-1.25, 1.25),
            np.random.uniform(0.06, 0.15),
            np.random.uniform(0.0, 10)
        ] for _ in range(n_wind)]
    else:
        wind = game_mgr.wind.copy()
        increment = [[
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.1, 0.1),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-2, 2)
        ] for _ in range(n_wind)]

        for i in range(n_wind):
            for j in range(4):
                wind[i][j] += increment[i][j]

            # Clamp values to domain bounds
            wind[i][0] = np.clip(wind[i][0], -1.9, 1.9)
            wind[i][1] = np.clip(wind[i][1], -1.25, 1.25)
            wind[i][2] = np.clip(wind[i][2], 0.06, 0.15)
            wind[i][3] = np.clip(wind[i][3], 0.0, 10.0)

    game_mgr.reset_wind()
    for w in wind:
        game_mgr.set_wind(w)

    avg_speed = np.mean([w[3] for w in wind]) if wind else 0.0

    if abs(avg_speed - old_avg_speed) > threshold:
        message["wind_speed"] = avg_speed
        message_changed = True
    else:
        message_changed = False

    return new_time, avg_speed, message_changed


if __name__ == "__main__":
    try:
        # === Setup socket for GUI communication ===
        # Create a server for socket communication
        host = '0.0.0.0'  # Use '127.0.0.1' to accept connections only from localhost
        port = 8888
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((host, port))
        s.listen()
        clients = []  # Track all client addresses
        print("Server waiting for connection...")
        while len(clients) < 1:  # !!! Wait for all client to connect
            conn, addr = s.accept()
            print("Connected by", addr)
            clients.append((conn, addr))  # Store the address
        print("All client connected")
        for conn, addr in clients:
            conn.setblocking(False)
        print('Set up socket for communication')

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
        n_wind = 3

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
        labeler = Labeler(spec)

        # === Allocator ===
        allocator = RandomAllocator(spec, agents_by_type, binding_manager, labeler)

        # === Simulation ===
        sim = Simulation(spec, ws, allocator, labeler)

        # === Busy airspace (wind) setup ===
        old_wind_average_speed = 0.0

        # === Survivior images ===
        victim_detected = set()
        victim_id = [0 for _ in range(n_targets)]
        victim_clicked = [0 for _ in range(n_targets)]
        victim_timing = [0.0 for _ in range(n_targets)]
        survivor_images = list(np.random.choice(range(1, 21), size=n_targets, replace=False))
        survivor_index = 0
        verify_response_pending = set()  # APs like p_verify_0_3_1_0 waiting for user
        victim_target_map = {}

        # === Message for GUI ===
        message = {'idx_image': None, 'tasks': tasks, 'wind_speed': None, 'progress': None, 'workload': None, 'vic_msg': None}
        for idx in range(len(clients)):
            clients[idx][0].sendall((json.dumps(message) + '\n').encode())
        recv_buffer = ''

        # === Initialize simulation time ===
        prev_time = time()
        init_time = time()
        wind_time = 0

        # The main loop for the GUI
        print('Main GUI initialized')
        while running:
            # === Avoid high CPU usage ===
            sleep(0.01)

            # === Message update ===
            data = None # Data received from the clients
            message = {'idx_image': None, 'tasks': None, 'wind_speed': None, 'progress': None, 'workload': None, 'vic_msg': None}

            # === Socket receive ===
            for conn, addr in clients:
                try:
                    chunk = conn.recv(4096).decode() # !!! Need to hear from all clients
                    if chunk:
                        recv_buffer += chunk
                        while '\n' in recv_buffer:
                            line, recv_buffer = recv_buffer.split('\n', 1)
                            if line.strip():
                                data = json.loads(line)
                                # print("Received data:", repr(data))
                except BlockingIOError:
                    pass

            # == Task priority update ===
            if data and data['tasks'] is not None:
                exist_task_idx = []
                for task in data['tasks']:
                    exist_task_idx.append(task['task_id'])
                    for j, ta in enumerate(tasks):
                        if ta[0] == task['task_id'] and ta[2] != task['priority']:
                            ta[2] = task['priority']
                            print(f'reset task {task["task_id"]} priority to {task["priority"]}')
                            break
                    
                message['tasks'] = tasks
                message_changed = True

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
            elif keys[pygame.K_s]:
                print(labeler.get_completed())

            # === Check if simulation is done ===
            if labeler.all_completed() and ws.all_mobile_agents_at_base():
                print(f"[t={running_time:02f}] Mission completed!")
                running = False
                landing = True

            # === Step the simulation ===
            if running_time < hover_duration:
                for drone in drones:
                    drone.takeoff_in_place(2.0)
            else:
                sim_outputs = sim.step(dt=dt, verbose=False)
                # Atomic propositions
                unlocked = sim_outputs["unlocked"]
                assignments = sim_outputs["assignments"]
                completed = sim_outputs["completed"]

                # === Victim detection after p_scan_i ===
                for ap in completed:
                    if ap.startswith("p_scan_") and ap not in victim_detected:
                        for agent in agents_by_type["drone"]:
                            if agent.has_completed(ap):
                                if agent.label.startswith("D"):
                                    drone_idx = int(agent.label[1:])
                                else:
                                    continue

                                victim_detected.add(ap)

                                image_id = survivor_images[survivor_index]
                                victim_id[drone_idx] = image_id
                                survivor_index += 1

                                # Store image_id -> target_id map
                                target_id = ap.split("_")[2]
                                victim_target_map[image_id] = target_id

                                message['idx_image'] = str(image_id)
                                pos_rounded = [round(coord, 2) for coord in agent.pos[:2]]
                                message['vic_msg'] = f'Drone {drone_idx + 1} finished scan at {pos_rounded}, please respond!'
                                victim_timing[drone_idx] = running_time
                                message_changed = True

                                # Block corresponding p_verify AP until human responds
                                verify_ap = f"p_verify_{target_id}_3_1_0"
                                verify_response_pending.add(verify_ap)
                                sim.verify_response_pending = verify_response_pending
                                # print("[DEBUG] victim_target_map:", victim_target_map)               

            # === GUI response handling ===
            if data and data['victim'] is not None:
                # print("[DEBUG] Received GUI victim input:", data.get('victim'))

                if victim_target_map.values():
                    target_id = next(iter(victim_target_map.values()))
                    image_id = next(iter(victim_target_map))
                    verify_ap = f"p_verify_{target_id}_3_1_0"
                    group_key = "_".join(verify_ap.split("_")[2:6])
                    
                    victim_target_map.pop(image_id, None)
                    print("[DEBUG] victim_target_map values:", list(victim_target_map.values()))

                    if data['victim'] == 'accept':
                        for idx in range(n_targets):
                            if victim_id[idx] == image_id:
                                victim_clicked[idx] = 1
                                break
                        group_key = str(target_id)
                        labeler.chosen_gate_per_group[group_key] = f"p_foundgate_{target_id}"
                        labeler._completed.add(verify_ap)
                        labeler.advance({verify_ap})

                    elif data['victim'] == 'reject':
                        for idx in range(n_targets):
                            if victim_id[idx] == image_id:
                                victim_clicked[idx] = 2
                                break
                        group_key = str(target_id)
                        labeler.chosen_gate_per_group[group_key] = f"p_notfoundgate_{target_id}"
                        labeler._completed.add(verify_ap)
                        labeler.advance({verify_ap})

                    elif data['victim'] == 'handover':
                        for idx in range(n_targets):
                            if victim_id[idx] == image_id:
                                victim_clicked[idx] = 3
                                break
                        group_key = str(target_id)
                        labeler.chosen_gate_per_group[group_key] = f"p_notfoundgate_{target_id}"
                        labeler._completed.add(verify_ap)
                        labeler.advance({verify_ap})

                data['victim'] = None

                # Clear drone-level GUI status after click is handled
                for idx in range(n_targets):
                    if victim_clicked[idx] > 0:
                        game_mgr.victim_clicked[idx] = 0
                        game_mgr.victim_id[idx] = 0
                        game_mgr.victim_detected[idx] = False
                        victim_clicked[idx] = 0
            
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

            # === Drone/GV status update ===
            for drone in drones:
                drone.health -= 0.01
                
            # === Awareness map ===
            pos_aware = [(d.position[0], d.position[1]) for d in drones]
            pos_aware = game_mgr.position_meter_to_gui(pos_aware)
            game_mgr.update_awareness(pos_aware, radius=40)

            # === Busy airspace ===
            if time() - wind_time > 3:
                wind_time, old_wind_average_speed, message_changed = update_wind(
                    game_mgr, wind_time, old_wind_average_speed, n_wind, message)
                
            # === Soket send ===
            # Decide which client to send the message!!!
            if message_changed:
                if message['tasks'] or message['wind_speed']:
                    # Send the message to all clients
                    for conn, addr in clients:
                        conn.sendall((json.dumps(message) + '\n').encode())
                else:
                    # Randomly select a client to send the message
                    selected_client = np.random.choice(range(len(clients)))
                    conn, addr = clients[selected_client]
                    conn.sendall((json.dumps(message) + '\n').encode())
                    message_changed = False

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
        
    finally:
        # Close the socket
        for conn, addr in clients:
            conn.close()
        s.close()
        pygame.quit()
        # Collect data
        print('Clean exit')