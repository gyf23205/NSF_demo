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
from ltl_core.specification import Specification, ENVIRONMENT_AP_PREFIXES
from ltl_core.binding_manager import BindingManager
from ltl_core.workspace import Workspace
from ltl_core.dag_builder import build_dag
from ltl_core.automaton_generator import compile_automata
from ltl_core.labeler import Labeler
from ltl_core.allocator import RandomAllocator
from ltl_core.simulation import Simulation
from ltl_core.visualization import draw_workspace

SLIDING_WINDOW = 60.0
grid_size = (50, 40)
# screen_size = (grid_size[0] * 30, grid_size[1] * 30)

# Event setup
FIREMSG_TIMES = [30.0, 45.0, 80.0, 110.0, 130.0, 160.0, 200.0]
SURVIVORMSG_TIMES = [25.0, 55.0, 70.0, 100.0, 120.0, 170.0, 210.0]
ATMMSG_TIMES = [35.0, 65.0, 90.0, 120.0, 150.0, 180.0, 220.0]

# Event variables
firemsg_idx = 0
survivormsg_idx = 0
atmmsg_idx = 0

# --- ATM message runtime ---
atm_prompt_id = 0
current_atm_prompt = None          # {"id", "coord": (x,y), "token", "text", "time"}
atm_sent_for_prompt = set()        # {prompt_id} already enqueued via buffers
atm_results = []                   # [True/False] per prompt in time 

# --- Survivor message runtime ---
surv_prompt_id = 0
current_surv_prompt = None      # {"id","text","correct"}; correct in {"Emergency","Serious","Minor"}
surv_sent_for_prompt = set()
surv_results = []               # True/False history

# --- Fire→Priority runtime ---
fire_prompt_id = 0
current_fire_prompt = None     # {"id","task_id","required","text","time"}
fire_sent_for_prompt = set()
fire_results = []              # True/False history


def _pick_coord_avoiding_targets(ws, grid_size=(50, 40)):
    """Pick a grid (x,y) not overlapping any target."""
    import random
    rows, cols = grid_size
    forbidden = set(tuple(t) for t in ws.target_locations)
    # optionally avoid base/hospital if you like:
    forbidden |= set(ws.base_area) | set(ws.hospital_area)
    rng = random.Random(314159)  # or use ws.rng for determinism
    candidates = [(x, y) for x in range(rows) for y in range(cols) if (x, y) not in forbidden]
    return rng.choice(candidates) if candidates else (rows // 2, cols // 2)


def _make_atm_prompt(x, y):
    token = f"NO_FLY_{x}_{y}"
    # keep text short, assertive; client will render red+bold
    text = f"Grid ({x}, {y}) is reserved for helicopter traffic. Keep out. Confirm by typing '{token}'."
    return token, text


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


def to_human_label(ap):
    # catch idle / no-assignment
    if not ap or ap == 'idle':
        return 'Idle'

    parts = ap.split('_')
    # prefix is e.g. "p_scan", "p_dropoff", etc.
    prefix = f"p_{parts[1]}"
    # try to parse the target index (and +1 for 1-based numbering)
    try:
        idx = int(parts[2]) + 1
    except:
        idx = None

    if prefix == 'p_nav':
        return f'Navigate {idx}'
    if prefix == 'p_scan':
        return f'Scan {idx}'
    if prefix == 'p_verify':
        return f'Verify {idx}'
    if prefix == 'p_pickup':
        return f'Pick up {idx}'
    if prefix == 'p_dropoff':
        return f'Drop off {idx}'
    if prefix == 'p_priority':
        return f'Set priority'
    if prefix == 'p_message':
        return f'Send Message'
    if prefix == 'p_nofly':
        return f'Set no fly'

    # environment APs: just humanize the prefix
    if prefix in ENVIRONMENT_AP_PREFIXES:
        # e.g. "p_firemsg" → "Fire message"
        label = prefix[2:].replace('msg', ' message').capitalize()
        return label

    # fallback
    return ap


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
        buffers['wind_speed'].append(avg_speed)

    return new_time, avg_speed


def compute_spiral_position(agent, dt, r_max=1.0, r_rate=0.02, angular_speed=0.6):
    """
    Update and return the agent's spiral (x, y) position for circling motion.

    Args:
        agent: object with .scan_center, .scan_radius, .scan_angle, .scan_time
        dt: timestep (seconds)
        r_max: maximum radius of spiral
        r_rate: radial growth rate (units/sec)
        angular_speed: angular velocity (radians/sec)

    Returns:
        np.array([x, y]) - new 2D position on spiral
    """

    # Time update
    agent.scan_time += dt

    # Radial distance grows linearly
    r = min(r_rate * agent.scan_time, r_max)

    # Angle increases
    agent.scan_angle = angular_speed * agent.scan_time

    # Compute position
    x = agent.scan_center[0] + r * np.cos(agent.scan_angle)
    y = agent.scan_center[1] + r * np.sin(agent.scan_angle)

    return np.array([x, y])


if __name__ == "__main__":
    try:
        # === Setup socket for GUI communication ===
        # Create a server for socket communication
        host = '127.0.0.1'  # Use '127.0.0.1' to accept connections only from localhost
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
        n_targets = 15
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
        ws = Workspace(size=grid_size, target_mask=s_mask, num_drones=n_drones, num_gvs=n_gvs, num_humans=n_humans, seed=120, margin=4)
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
        humans = ws.agents["humans"]
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
        for human in ws.agents["humans"]:
            human.util_history = [(0.0, 'idle')]
            human.last_state   = 'idle'
            human.utilization  = 0

        # === GameMgr for rendering ===
        game_mgr = GameMgr(drones, gvs, humans, ws)

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

        # === Human agents: new fields ===
        for human in ws.agents["humans"]:
            human.util_history = []         # List of (t, state) where state \in {busy, idle}
            human.last_state = None         # Track for edge detection

        # === Busy airspace (wind) setup ===
        old_wind_average_speed = 0.0

        # === Survivior images ===
        victim_detected = set()
        victim_id = [0 for _ in range(n_targets)]
        victim_clicked = [0 for _ in range(n_targets)]
        victim_timing = [0.0 for _ in range(n_targets)]
        survivor_images = list(np.random.choice(range(1, 596), size=n_targets, replace=False))
        survivor_index = 0
        verify_response_pending = set()  # APs like p_verify_0_3_1_0 waiting for user
        
        # === Verify gating (single‑user routing) ===
        pending_images = {}         # target_id (str) -> image_id (int), waiting to be sent
        sent_for_target = set()     # target_ids already sent to user (avoid dup)
        active_target_for_user = None

        # === Message for GUI ===
        message = {'idx_image': None, 'tasks': tasks, 'wind_speed': None, 'progress': None, 'workload': None, 'vic_msg': None}
        for idx in range(len(clients)):
            clients[idx][0].sendall((json.dumps(message) + '\n').encode())
        recv_buffer = ''

        # === Initialize simulation time ===
        prev_time = time()
        init_time = time()
        wind_time = 0

        # === Working area: temporary ===
        prev_assignments = []

        # The main loop for the GUI
        print('Main GUI initialized')

        buffers = {
            'idx_image':[],
            'tasks':[],
            'wind_speed':[],
            'progress':[],
            'workload':[],
            'vic_msg':[],
            'atm_prompt':[],   # server → one client: {"id","text","token","coord":[x,y]}
            'atm_clear':[],    # server → one client: {"id": ...}
            'surv_prompt': [],   # server→one: {"id","text","choices":["Emergency","Serious","Minor"]}
            'surv_clear':  [],   # server→one: {"id":...,"ok":True/False}
            'fire_prompt': [],   # server→one: {"id","task_id","text","required"}
            'fire_clear':  [],   # server→one: {"id","ok":True|False,"reason":...}
            }
        while running:
            # === Avoid high CPU usage ===
            sleep(0.01)

            # === Message update ===
            data = None # Data received from the clients

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
            if isinstance(data, dict) and data.get('tasks') is not None:
                exist_task_idx = []
                for task in data['tasks']:
                    exist_task_idx.append(task['task_id'])
                    for j, ta in enumerate(tasks):
                        if ta[0] == task['task_id'] and ta[2] != task['priority']:
                            ta[2] = task['priority']
                            print(f'reset task {task["task_id"]} priority to {task["priority"]}')
                            break
                
                buffers['tasks'].append(tasks)
                # message['tasks'] = tasks
                # message_changed = True

            # === Compute timestep ===
            current_time = time()
            dt = current_time - prev_time
            prev_time = current_time
            running_time = current_time - init_time

            # === Event handling ===
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print(f"[t={running_time:.2f}] Quit by window close.")
                    running = False
                    landing = False

            # === Key handling ===
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE]:
                print(f"[t={running_time:.2f}] Escape by keyboard.")
                running = False
                landing = True
            elif keys[pygame.K_s]:
                print(labeler.get_completed())

            # === Check if simulation is done ===
            if labeler.all_completed() and ws.all_mobile_agents_at_base():
                print(f"[t={running_time:.2f}] Mission completed!")
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

                # === Working area: temporary ===
                if assignments != prev_assignments:
                    print(f"Assigned: {[f'{a.label}→{ap}' for a, ap in assignments.items()]}")
                    prev_assignments = assignments

                # === Agent: assigned function ===
                # 1) update the symbolic Agents
                for agent in ws.get_all_agents():
                    agent.status = assignments.get(agent, 'Idle')
                # 2) mirror it onto the visuals so GameMgr can see it
                for sym, visual in agent_to_visual.items():
                    raw_ap = assignments.get(sym)
                    human_ap = to_human_label(raw_ap)
                    sym.status = human_ap
                    visual.status = human_ap

                # === Human assignment history & utilization ===
                now = running_time
                for human in ws.agents["humans"]:
                    # 1) Detect busy vs idle (busy if they _have_ an assignment)
                    assigned = assignments.get(human)
                    new_state = 'busy' if assigned is not None else 'idle'

                    # 2) On state‐change, record the timestamp
                    if new_state != human.last_state:
                        human.util_history.append((now, new_state))
                        human.last_state = new_state

                    # 3) Prune old events outside the sliding window
                    human.util_history = [
                        (t,s) for t,s in human.util_history
                        if now - t <= SLIDING_WINDOW
                    ]

                    # 4) Compute % utilization over the last window
                    human.utilization = compute_utilization(human, now, SLIDING_WINDOW)

                # === Monitor APs tiggers
                # 1. possible new emergency events (FIRE → ask human to set priority)
                if (firemsg_idx < len(FIREMSG_TIMES)
                        and running_time >= FIREMSG_TIMES[firemsg_idx]):
                    print(f"[t={running_time:.1f}] Triggering new fire message")
                    labeler.advance({"p_firemsg_0_0_0_0"})
                    firemsg_idx += 1

                    # Create a priority prompt tied to an AVAILABLE task id
                    available_task_ids = [t[0] for t in tasks]  # 1-based task ids
                    if available_task_ids:
                        chosen_task_id = int(np.random.choice(available_task_ids))
                        fire_prompt_id += 1
                        required = 2  # "HIGH" in your 0/1/2 scale
                        text = f"Region {chosen_task_id} is in danger. Set its priority to HIGH (2)."
                        current_fire_prompt = {
                            "id": fire_prompt_id,
                            "task_id": chosen_task_id,
                            "required": required,
                            "text": text,
                            "time": running_time,
                        }
                        fire_sent_for_prompt.clear()

                # 2. possible new survivor messages
                if (survivormsg_idx < len(SURVIVORMSG_TIMES)
                        and running_time >= SURVIVORMSG_TIMES[survivormsg_idx]):
                    print(f"[t={running_time:.1f}] Triggering new survivor message")

                    import random
                    scenario = random.choice([1, 2, 3])
                    if scenario == 1:
                        surv_text = "Survivor message: the survivor is unconscious and vitals are poor."
                        surv_correct = "Emergency"
                    elif scenario == 2:
                        surv_text = 'Survivor message: "I am seriously hurt and need urgent care at the hospital."'
                        surv_correct = "Serious"
                    else:
                        surv_text = 'Survivor message: "My pain is minor."'
                        surv_correct = "Minor"

                    surv_prompt_id += 1
                    current_surv_prompt = {"id": surv_prompt_id, "text": surv_text, "correct": surv_correct}
                    surv_sent_for_prompt.clear()

                    # ENV-driven AP holds now
                    labeler.advance({"p_survivormsg_0_0_0_0"})
                    survivormsg_idx += 1

                # 3. ATM scheduler (prompt + env broadcast)
                if (atmmsg_idx < len(ATMMSG_TIMES)) and (running_time >= ATMMSG_TIMES[atmmsg_idx]):
                    from math import isfinite
                    # global atm_prompt_id, current_atm_prompt, atm_sent_for_prompt

                    # If previous prompt is still pending, record failure and replace it
                    if current_atm_prompt is not None:
                        print(f"[t={running_time:.1f}] ATM overrun: previous prompt {current_atm_prompt['id']} failed")
                        atm_results.append(False)

                    # Create a new prompt
                    x, y = _pick_coord_avoiding_targets(ws, grid_size=grid_size)
                    token, text = _make_atm_prompt(x, y)
                    atm_prompt_id += 1
                    current_atm_prompt = {"id": atm_prompt_id, "coord": (x, y), "token": token, "text": text, "time": running_time}
                    atm_sent_for_prompt.clear()

                    # Add a small no-fly dot to the environment so path planners can avoid it
                    # (we'll handle planner-side propagation in rrt_connect_ltl / utils later)
                    try:
                        ws.env.obs_circle.append([x, y, 0.8])
                    except Exception as e:
                        print(f"[ATM] Warning: failed to add no-fly dot: {e}")

                    # Environment AP broadcast occurs at the scheduled tick
                    print(f"[t={running_time:.1f}] ATM broadcast: {token} @ ({x},{y})")
                    labeler.advance({"p_atmmsg_0_0_0_0"})

                    atmmsg_idx += 1

                # === Victim detection after p_scan_i ===
                for ap in completed:
                    if ap.startswith("p_scan_") and ap not in victim_detected:
                        for agent in agents_by_type["drone"]:
                            if agent.has_completed(ap):
                                drone_idx = int(agent.label[1:])
                                victim_detected.add(ap)

                                image_id = survivor_images[survivor_index]
                                survivor_index += 1
                                print(f"time: {running_time}, survivor_index: {survivor_index}, AP: {ap}, Image: {image_id}")

                                # Keep for later; only send when p_verify_* is actually assigned
                                target_id = ap.split("_")[2]             # str
                                pending_images[target_id] = image_id

                                # Only one AP considered for each drone
                                break

                # === If a human is assigned p_verify_*, send the pending image now ===
                for agent, ap in assignments.items():
                    if isinstance(ap, str) and ap.startswith("p_verify_") and str(getattr(agent, "role", "")).startswith("human"):
                        tid = ap.split("_")[2]  # target id as string
                        # only send once per target, and only if we actually have an image
                        if tid in pending_images and tid not in sent_for_target:
                            img_id = pending_images[tid]
                            # send both the image and the tid
                            buffers['idx_image'].append({"image_id": int(img_id), "tid": tid})
                            buffers['vic_msg'].append(f'Verify target {int(tid)+1}: please respond')
                            sent_for_target.add(tid)
                            # (optional) keep active_target_for_user as a fallback, but no longer relied on
                            active_target_for_user = tid

                # === If a human is assigned p_nofly_*, send the ATM prompt now ===
                for agent, ap in assignments.items():
                    if isinstance(ap, str) and ap.startswith("p_nofly_") and str(getattr(agent, "role", "")).startswith("human"):
                        if current_atm_prompt and (current_atm_prompt["id"] not in atm_sent_for_prompt):
                            buffers['atm_prompt'].append({
                                "id": current_atm_prompt["id"],
                                "text": current_atm_prompt["text"],     # client renders red+bold
                                "token": current_atm_prompt["token"],
                                "coord": list(current_atm_prompt["coord"]),
                            })
                            atm_sent_for_prompt.add(current_atm_prompt["id"])

                # === If a human is assigned p_message_*, send the survivor prompt now ===
                for agent, ap in assignments.items():
                    if isinstance(ap, str) and ap.startswith("p_message_") and str(getattr(agent, "role", "")).startswith("human"):
                        if current_surv_prompt and (current_surv_prompt["id"] not in surv_sent_for_prompt):
                            buffers['surv_prompt'].append({
                                "id": current_surv_prompt["id"],
                                "text": current_surv_prompt["text"],
                                "choices": ["Emergency", "Serious", "Minor"]
                            })
                            surv_sent_for_prompt.add(current_surv_prompt["id"])

                # === If a human is assigned p_priority_*, send the fire/priority prompt now ===
                for agent, ap in assignments.items():
                    if isinstance(ap, str) and ap.startswith("p_priority_") and str(getattr(agent, "role", "")).startswith("human"):
                        if current_fire_prompt and (current_fire_prompt["id"] not in fire_sent_for_prompt):
                            buffers['fire_prompt'].append({
                                "id": current_fire_prompt["id"],
                                "task_id": current_fire_prompt["task_id"],
                                "text": current_fire_prompt["text"],
                                "required": current_fire_prompt["required"],
                            })
                            fire_sent_for_prompt.add(current_fire_prompt["id"])

            # === GUI response handling ===
            if isinstance(data, dict) and data.get('victim') is not None:
                # respond to the specific target we sent to the user
                if active_target_for_user is not None:
                    target_id = active_target_for_user
                    image_id = pending_images.get(target_id)
                    verify_ap = f"p_verify_{target_id}_3_1_0"
                    idx = int(target_id)
            
            if isinstance(data, dict) and data.get('victim') is not None:
                # prefer the tid the GUI tells us it acted on
                target_id = data.get('verify_tid', None)
                if target_id is None:
                    # fallback to the last-active if old client, but new client will set verify_tid
                    target_id = active_target_for_user
                if target_id is not None:
                    image_id = pending_images.get(target_id)
                    verify_ap = f"p_verify_{target_id}_3_1_0"
                    idx = int(target_id)

                    # choose gate
                    if data['victim'] == 'accept':
                        labeler.chosen_gate_per_group[target_id] = f"p_foundgate_{target_id}"
                    elif data['victim'] in ('reject', 'handover'):
                        labeler.chosen_gate_per_group[target_id] = f"p_notfoundgate_{target_id}"

                    chosen_gate = labeler.chosen_gate_per_group.get(target_id)
                    if chosen_gate:
                        labeler.advance({verify_ap, chosen_gate})
                    else:
                        labeler.advance({verify_ap})

                    # bookkeep
                    if image_id is not None:
                        victim_id[idx] = image_id
                    victim_timing[idx] = running_time

                    # remove from task list
                    tasks = [item for item in tasks if item[0] != idx + 1]
                    buffers['tasks'].append(tasks)
                    game_mgr.task = [item for item in game_mgr.task if item[0] != idx + 1]

                    # clear this target’s pending state
                    pending_images.pop(target_id, None)
                    sent_for_target.discard(target_id)
                    active_target_for_user = None

                data['victim'] = None
            
            # === Auto-fail if the chosen task disappears while prompt is active ===
            if current_fire_prompt:
                current_ids = [t[0] for t in tasks]
                if current_fire_prompt["task_id"] not in current_ids:
                    fire_results.append(False)
                    labeler.advance({"p_priority_0_3_1_0"})
                    buffers['fire_clear'].append({
                        "id": current_fire_prompt["id"],
                        "ok": False,
                        "reason": "task_gone"
                    })
                    current_fire_prompt = None
                    fire_sent_for_prompt.clear()

            # === ATM reply handling ===
            if isinstance(data, dict) and data.get('atm_reply') is not None:
                payload = data['atm_reply']
                if current_atm_prompt and payload.get("id") == current_atm_prompt["id"]:
                    typed = payload.get("typed", "")
                    token = current_atm_prompt["token"]
                    if isinstance(typed, str) and typed.startswith(token):
                        atm_results.append(True)
                        print(f"[t={running_time:.1f}] ATM confirmation accepted: {token}")
                        # mark the human action complete
                        labeler.advance({"p_nofly_0_3_1_0"})
                        buffers['atm_clear'].append({"id": current_atm_prompt["id"]})
                        current_atm_prompt = None
                        atm_sent_for_prompt.clear()

            # === Survivor reply handling (top-right buttons) ===
            if isinstance(data, dict) and data.get('surv_reply') is not None:
                payload = data['surv_reply']   # {"id":..., "choice":"Emergency|Serious|Minor"}
                if current_surv_prompt and payload.get("id") == current_surv_prompt["id"]:
                    choice = payload.get("choice")
                    ok = (choice == current_surv_prompt["correct"])
                    surv_results.append(ok)
                    if ok:
                        # Human action AP after correct triage
                        labeler.advance({"p_message_0_3_1_0"})
                        buffers['surv_clear'].append({"id": current_surv_prompt["id"], "ok": True})
                        current_surv_prompt = None
                        surv_sent_for_prompt.clear()
                    else:
                        # keep prompt active, nudge client if you want
                        buffers['surv_clear'].append({"id": current_surv_prompt["id"], "ok": False})
            
            # === Fire→Priority reply handling (right-bottom numeric input) ===
            if isinstance(data, dict) and data.get('priority_reply') is not None:
                payload = data['priority_reply']  # {"id":..., "task_id": int, "priority": int}
                if current_fire_prompt and payload.get("id") == current_fire_prompt["id"]:
                    tid = int(payload.get("task_id"))
                    pr  = payload.get("priority")

                    # Check availability of the task *now*
                    available_ids = [t[0] for t in tasks]
                    if tid not in available_ids:
                        # Task disappeared before reply → FAIL & advance
                        fire_results.append(False)
                        labeler.advance({"p_priority_0_3_1_0"})
                        buffers['fire_clear'].append({"id": current_fire_prompt["id"], "ok": False, "reason": "task_gone"})
                        current_fire_prompt = None
                        fire_sent_for_prompt.clear()
                    else:
                        # Validate priority
                        if isinstance(pr, int) and pr == current_fire_prompt["required"]:
                            # Update the task table priority
                            for ta in tasks:
                                if ta[0] == tid:
                                    ta[2] = pr
                                    break
                            buffers['tasks'].append(tasks)

                            fire_results.append(True)
                            labeler.advance({"p_priority_0_3_1_0"})
                            buffers['fire_clear'].append({"id": current_fire_prompt["id"], "ok": True})
                            current_fire_prompt = None
                            fire_sent_for_prompt.clear()
                        else:
                            # Keep prompt active; optional nudge
                            buffers['fire_clear'].append({"id": current_fire_prompt["id"], "ok": False, "reason": "wrong_value"})
            
            # === Drone/GV positions ===
            for agent, visual in agent_to_visual.items():
                pos = ws.grid_to_game_mgr(agent.pos)
                if hasattr(agent, "role") and agent.role == "drones":
                    # Drone circling movement for p_scan_i
                    task = getattr(agent, "current_symbolic_task", None)
                    if isinstance(task, str) and task.startswith("p_scan"):
                        # Initialize scan parameters
                        if getattr(agent, "last_scan_ap", None) != task:
                            # print(f"time = {int(running_time)}, task = {task}, drone = {agent.label}")
                            agent.scan_center = np.copy(pos)
                            agent.scan_time = 0.0
                            agent.scan_angle = 0.0
                            agent.last_scan_ap = task  # track currently handled scan
                        # Compute new position on spiral
                        pos = compute_spiral_position(agent, dt)
                        agent.pos = ws.game_mgr_to_grid(pos)
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
                wind_time, old_wind_average_speed = update_wind(
                    game_mgr, wind_time, old_wind_average_speed, n_wind, message)
                
            # === Soket send ===
            # Decide which client to send the message!!!
            if any(len(v) > 0 for v in buffers.values()):
                message_all = {'tasks': None, 'wind_speed': None}
                message_one = {'idx_image': None, 'vic_msg': None, 'workload': None,
                               'atm_prompt': None, 'atm_clear': None,
                               'surv_prompt': None, 'surv_clear': None,
                               'fire_prompt': None, 'fire_clear': None}

                if buffers['idx_image']:
                    message_one['idx_image'] = buffers['idx_image'].pop(0)

                if buffers['tasks']:
                    message_all['tasks'] = buffers['tasks'].pop(0)

                if buffers['wind_speed']:
                    message_all['wind_speed'] = buffers['wind_speed'].pop(0)

                # if buffers['progress']:
                #     message['progress'] = buffers['progress'].pop(0)

                if buffers['workload']:
                    message_one['workload'] = buffers['workload'].pop(0)

                if buffers['vic_msg']:
                    message_one['vic_msg'] = buffers['vic_msg'].pop(0)

                # ATM prompt/clear
                if buffers['atm_prompt']:
                    message_one['atm_prompt'] = buffers['atm_prompt'].pop(0)
                if buffers['atm_clear']:
                    message_one['atm_clear'] = buffers['atm_clear'].pop(0)

                if buffers['surv_prompt']:
                    message_one['surv_prompt'] = buffers['surv_prompt'].pop(0)
                if buffers['surv_clear']:
                    message_one['surv_clear'] = buffers['surv_clear'].pop(0)

                if buffers['fire_prompt']:
                    message_one['fire_prompt'] = buffers['fire_prompt'].pop(0)
                if buffers['fire_clear']:
                    message_one['fire_clear'] = buffers['fire_clear'].pop(0)

                # Send messages to clients
                if any(v is not None for v in message_all.values()):
                    for conn, addr in clients:
                        conn.sendall((json.dumps(message_all) + '\n').encode())
                if any(v is not None for v in message_one.values()):
                    selected_client = np.random.choice(range(len(clients)))
                    conn, addr = clients[selected_client]
                    conn.sendall((json.dumps(message_one) + '\n').encode())

            # === Draw GUI: simple ===
            # draw_workspace(screen, ws, screensize=screen_size)

            # === Planned path ===
            game_mgr.paths = []
            for agent, visual in agent_to_visual.items():
                if len(agent.path) > 1:
                    # convert each (x, y) in grid coords to pixel coords
                    gui_pts = [
                        ws.grid_to_pixel(tuple(p), grid_size=grid_size, screen_size=(900, 720))
                        for p in agent.path
                    ]
                    # choose a color per type
                    col = (240, 128, 128) if agent.role == 'drones' else (255, 255, 150)
                    game_mgr.paths.append((col, gui_pts))

            # === Draw GUI: Render the game manager ===
            game_mgr.render(vor, centroids)

        # === Drone landing ===
        game_mgr.paths = [] # Clear path
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