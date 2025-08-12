import sys
sys.path.append('C:/Users/sooyung/Research/NSF_demo')
import pygame
import csv, random
from time import time, sleep, strftime
from gui_panel_ltl import GameMgr
from vehicles import VirtualDrone, VirtualGV
from scipy.spatial import Voronoi
import numpy as np
import socket
import json
import ctypes
from ltl_core.specification import Specification, ENVIRONMENT_AP_PREFIXES, AP_TYPE_PREFIX_MAP, get_ap_prefix
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

# Event setup: TO DO: They should be time + progress based triggers
# FIREMSG_TIMES = [30.0, 45.0, 80.0, 110.0]
# SURVIVORMSG_TIMES = [25.0, 55.0, 70.0, 100.0]
# ATMMSG_TIMES = [35.0, 65.0, 90.0, 120.0]
FIREMSG_TIMES = [30.0, 45.0, 80.0, 110.0, 135.0, 150.0, 200.0, 250.0, 300.0]
SURVIVORMSG_TIMES = [25.0, 55.0, 70.0, 100.0, 140.0, 156.0, 169.3, 205.0, 264.5]
ATMMSG_TIMES = [35.0, 65.0, 90.0, 105.0, 120.0, 154.2, 185.3, 210.4, 250.4, 290.0]
# FIREMSG_TIMES = [80.0]
# SURVIVORMSG_TIMES = [100.0]
# ATMMSG_TIMES = [120.0]

# Drone latency
L_MIN, L_MAX = 40.0, 500.0      # realistic latency bounds (ms)
EMA = 0.85                      # smoothing (higher = smoother)

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
surv_credit = 0                # how many “p_verify … completed” since last triage
credited_verifies = set()      # which verify APs we've already counted

# --- Fire→Priority runtime ---
fire_prompt_id = 0
current_fire_prompt = None     # {"id","task_id","required","text","time"}
fire_sent_for_prompt = set()
fire_results = []              # True/False history
fire_time_credit = 0

# --- Task removal after pickup gating ---
pickup_cleared = set()


def _init_mixer():
    if not pygame.mixer.get_init():
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    pygame.mixer.set_num_channels(8)


def _tone(freq=660, dur=0.15, vol=0.5, sr=44100):
    """Return a pygame Sound of a single sine tone."""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    wave = (np.sin(2 * np.pi * freq * t) * (32767 * vol)).astype(np.int16)
    stereo = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(stereo.copy())


def set_always_on_top(enable=True):
    """Make the current Pygame window topmost (Windows only)."""
    try:
        hwnd = pygame.display.get_wm_info().get("window")
    except Exception:
        hwnd = None
    if not hwnd:
        return  # window not created yet

    HWND_TOPMOST = -1
    HWND_NOTOPMOST = -2
    SWP_NOMOVE = 0x0002
    SWP_NOSIZE = 0x0001
    SWP_SHOWWINDOW = 0x0040

    ctypes.windll.user32.SetWindowPos(
        hwnd,
        HWND_TOPMOST if enable else HWND_NOTOPMOST,
        0, 0, 0, 0,
        SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW
    )


def _atm_region_id(pid: int) -> str:
    return f"ATM_{pid}"


def _ap_target_id(ap: str):
    try:
        parts = ap.split('_')
        return int(parts[2])  # 0-based target id in your APs
    except Exception:
        return None


def _pick_coord_avoiding_targets(ws, grid_size=(50, 40)):
    """Pick a grid (x,y) not overlapping any target."""
    import random
    rows, cols = grid_size
    forbidden = set(tuple(t) for t in ws.target_locations)
    # optionally avoid base/hospital if you like:
    forbidden |= set(ws.base_area) | set(ws.hospital_area)
    rng = random.Random()  # or use ws.rng for determinism
    candidates = [(x, y) for x in range(rows) for y in range(cols) if (x, y) not in forbidden]
    return rng.choice(candidates) if candidates else (rows // 2, cols // 2)


def compute_utilization(human, now, window=SLIDING_WINDOW):
    events = sorted(human.util_history, key=lambda x: x[0])
    t0 = now - window

    # Determine state at t0 (last event at or before t0)
    state_at_t0 = 'idle'
    for t, s in events:
        if t <= t0:
            state_at_t0 = s
        else:
            break

    busy_time = 0.0
    prev_t, prev_s = t0, state_at_t0

    # Integrate only over [t0, now]
    for t, s in events:
        if t < t0:
            continue
        if prev_s == 'busy':
            busy_time += t - prev_t
        prev_t, prev_s = t, s

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
    if prefix == 'p_triage':
        return f'Triage'
    if prefix == 'p_atmconfirm':
        return f'ATM confirm'

    # environment APs: just humanize the prefix
    if prefix in ENVIRONMENT_AP_PREFIXES:
        # e.g. "p_firemsg" → "Fire message"
        label = prefix[2:].replace('msg', ' message').capitalize()
        return label

    # fallback
    return ap


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
        n_drones = 3
        n_gvs = 4
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

        # === Load symptoms ===
        triage_symptoms = {1:[], 2:[], 3:[]}
        with open('examples/data/triage_symptoms.csv', 'r', newline='') as f:  # put the file next to your script or adjust path
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lab = int(row['label'])
                    triage_symptoms[lab].append(row['symptom'])
                except Exception:
                    pass

        # === Load ATM message templates (token,sentence) ===
        atm_rows = []  # list of (token_tmpl, sentence_tmpl)
        with open('examples/data/atm_messages.csv', 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tok = (row.get('token') or '').strip()       # e.g. "NO_FLY_{x}_{y}"
                sent = (row.get('sentence') or '').strip()   # e.g. "Grid ({x}, {y}) ... '{token}'."
                if tok and sent:
                    atm_rows.append((tok, sent))
        if not atm_rows:
            raise RuntimeError("No ATM templates loaded from data/atm_messages.csv")

        # === Agents and Binding Manager ===
        agents_by_type = {
            "drones": ws.agents["drones"],
            "gvs": ws.agents["gvs"],
            "humans": ws.agents["humans"]
        }
        binding_manager.agents_by_type = agents_by_type

        # === Create visual agents from symbolic agents ===
        drones = []
        gvs = []
        humans = ws.agents["humans"]
        agent_to_visual = {}

        for i, agent in enumerate(agents_by_type["drones"]):
            pos = agent.pos
            if len(pos) == 2:
                pos = np.append(pos, 0.0)  # Add dummy altitude
            vd = VirtualDrone(i, tuple(pos))
            vd.latency_ms = 110
            drones.append(vd)
            agent_to_visual[agent] = vd
        for i, agent in enumerate(agents_by_type["gvs"]):
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
        set_always_on_top(True)

        # === Get drone start positions in GUI space ===
        takeoff_positions = [agent.pos[:2] for agent in agents_by_type["drones"]]
        takeoff_gui = [ws.grid_to_pixel(pos, grid_size=(50, 40), screen_size=(1125, 900)) for pos in takeoff_positions]
        game_mgr.set_takeoff_positions(takeoff_gui)

        # === Task interpretation ===
        tasks = []
        for idx, pos in enumerate(ws.target_locations):
            gui_pos = ws.grid_to_pixel(pos, grid_size=(50, 40), screen_size=(1125, 900))
            tasks.append([idx + 1, list(gui_pos), 0, None, None, [int(pos[0]), int(pos[1])]])
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

        # Play short beep
        _init_mixer()
        _tone(660, 0.4, 0.8).play()

        buffers = {
            'idx_image':[],
            'tasks':[],
            'wind_speed':[],
            'progress':[],
            'workload':[],
            'vic_msg':[],
            'atm_prompt':[],   # server → one client: {"id","text","token","coord":[x,y]}
            'surv_prompt': [],   # server→one: {"id","text","choices":["Emergency","Serious","Minor"]}
            'fire_prompt': [],   # server→one: {"id","task_id","text","required"}
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

                            # Complete the fire-priority prompt only if this edit satisfies it
                            if current_fire_prompt and task['task_id'] == current_fire_prompt.get('task_id'):
                                def _as_int(v):
                                    try:
                                        # handles int, str like "2", and numpy ints
                                        return int(str(v).strip())
                                    except Exception:
                                        return None

                                new_pr = _as_int(task['priority'])
                                req_pr = _as_int(current_fire_prompt.get('required'))

                                if new_pr is not None and req_pr is not None and new_pr == req_pr:
                                    fire_results.append(True)
                                    labeler.advance({"p_priority_0_3_1_0"})
                                    # only clear the highlight when the requirement is met
                                    game_mgr.special_regions.remove_region(ta[0])
                                    current_fire_prompt = None
                                    fire_sent_for_prompt.clear()
                                # else: leave the region and prompt active

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

                # === Remove task rows once p_pickup_*_2_1_0 is completed ===
                removed_any = False
                for ap in completed:
                    if isinstance(ap, str) and ap.startswith("p_pickup_") and ap.endswith("_2_1_0"):
                        try:
                            tid0 = int(ap.split("_")[2])   # 0-based target id in the AP
                        except (ValueError, IndexError):
                            continue
                        if tid0 in pickup_cleared:
                            continue  # already removed
                        pickup_cleared.add(tid0)

                        tid1 = tid0 + 1  # your task table uses 1-based ids
                        # Remove from both server-side task list and GameMgr copy
                        tasks = [row for row in tasks if row[0] != tid1]
                        game_mgr.task = [row for row in game_mgr.task if row[0] != tid1]
                        removed_any = True
                if removed_any:
                    # Notify clients once per tick with the updated table
                    buffers['tasks'].append(tasks[:])  # shallow copy is fine

                # Build "currently assigned" maps from assignments → {tid: idx}
                assigned_drone_by_tid = {}
                assigned_gv_by_tid = {}

                for agent, ap in assignments.items():
                    if not isinstance(ap, str):
                        continue
                    tid0 = _ap_target_id(ap)  # 0-based
                    if tid0 is None:
                        continue
                    role = getattr(agent, "role", None)
                    if role == "drones":
                        # agent.label like 'D0' → 0
                        try:
                            assigned_drone_by_tid[tid0] = int(agent.label[1:])
                        except Exception:
                            assigned_drone_by_tid[tid0] = None
                    elif role == "gvs":
                        try:
                            assigned_gv_by_tid[tid0] = int(agent.label[1:])
                        except Exception:
                            assigned_gv_by_tid[tid0] = None

                # Update the tasks table to reflect current assignments (None if not assigned)
                for row in tasks:
                    target_id_1b = row[0]
                    tid0 = target_id_1b - 1
                    row[3] = assigned_drone_by_tid.get(tid0, None)  # assigned_drone
                    row[4] = assigned_gv_by_tid.get(tid0, None)     # assigned_gv

                # Whenever assignments change, also push the refreshed task table to the client
                if assignments != prev_assignments:
                    import copy
                    buffers['tasks'].append(copy.deepcopy(tasks))

                # === Working area: temporary ===
                if assignments != prev_assignments:
                    print(f"Assigned: {[f'{a.label}→{ap}' for a, ap in assignments.items()]}")
                    prev_assignments = assignments

                # === Agent: assigned function ===
                # 1) update the symbolic Agents
                for agent in ws.get_all_agents():
                    raw_ap = assignments.get(agent)
                    agent.status = to_human_label(raw_ap)
                # 2) mirror it onto the visuals so GameMgr can see it
                for sym, visual in agent_to_visual.items():
                    raw_ap = assignments.get(sym)
                    human_ap = to_human_label(raw_ap)
                    sym.status = human_ap
                    visual.status = human_ap
                    # Carrying = True only during dropoff; False otherwise
                    if getattr(sym, "role", "") == "gvs":
                        visual.carrying = bool(isinstance(raw_ap, str) and raw_ap.startswith("p_dropoff_"))

                # === Drone latency update (EMA + noise; uses spec.AP_TYPE_PREFIX_MAP) ===
                for sym, visual in agent_to_visual.items():
                    if getattr(sym, "role", "") != "drones":
                        continue

                    ap = assignments.get(sym, None)         # e.g., 'p_scan_0_1_1_0'
                    # default to moderate if idle/unknown
                    target_mu, target_sigma = 110.0, 15.0

                    if isinstance(ap, str):
                        prefix = get_ap_prefix(ap)          # e.g., 'p_scan'
                        ap_type = AP_TYPE_PREFIX_MAP.get(prefix)  # 'physical'|'symbolic'|None
                        if ap_type == "symbolic":
                            target_mu, target_sigma = 350.0, 30.0   # high during symbolic
                        elif ap_type == "physical":
                            target_mu, target_sigma = 110.0, 15.0   # moderate during physical

                    sample = float(np.random.normal(target_mu, target_sigma))
                    sample = max(L_MIN, min(L_MAX, sample))         # clamp
                    prev = getattr(visual, "latency_ms", 110.0)
                    visual.latency_ms = EMA * prev + (1.0 - EMA) * sample

                # === Human assignment history & utilization ===
                now = running_time
                for human in ws.agents["humans"]:
                    # 1) Detect busy vs idle (busy if they _have_ an assignment)
                    assigned = assignments.get(human)
                    new_state = 'busy' if assigned is not None else 'idle'

                    # ensure history is seeded once
                    if not getattr(human, "util_history", None):
                        human.util_history = [(0.0, 'idle')]
                        human.last_state = 'idle'

                    # 2) On state‐change, record the timestamp
                    if new_state != human.last_state:
                        human.util_history.append((now, new_state))
                        human.last_state = new_state

                    # 3) Prune old events, but KEEP the last event before t0 so we know the state at window start
                    t0 = now - SLIDING_WINDOW
                    ev = human.util_history
                    older = [e for e in ev if e[0] < t0]
                    newer = [e for e in ev if e[0] >= t0]
                    keep = ([max(older, key=lambda x: x[0])] if older else []) + newer
                    human.util_history = keep

                    # 4) Compute % utilization over the last window
                    human.utilization = compute_utilization(human, now, SLIDING_WINDOW)

                # === Monitor APs tiggers
                # 1-1. possible new emergency events (FIRE → ask human to set priority)
                if (firemsg_idx < len(FIREMSG_TIMES)) and (running_time >= FIREMSG_TIMES[firemsg_idx]):
                    fire_time_credit += 1
                    firemsg_idx += 1

                # 1-2. If we have a credit and at least one remaining task, create a prompt now
                if (fire_time_credit > 0) and (current_fire_prompt is None) and (len(tasks) > 0):
                    # available_task = [t for t in tasks]  # 1-based ids still on table
                    if tasks:
                        chosen_task = int(np.random.choice(len(tasks)))
                        chosen_task_id = int(tasks[chosen_task][0])
                        chosen_task_pos = tasks[chosen_task][1]
                        required = int(np.random.choice([1, 2]))
                        current_priority = tasks[chosen_task][2]

                        if int(current_priority) == required:
                            pass
                        else:
                            fire_prompt_id += 1
                            game_mgr.special_regions.add_region(chosen_task_id, chosen_task_pos, 50, required)
                            # text = f"Region {chosen_task_id} is in danger. Set its priority to HIGH (2)."
                            text = "Fire is approaching to one of rescue regions. Set priority!"
                            current_fire_prompt = {
                                "id": int(fire_prompt_id),
                                "task_id": chosen_task_id,
                                "required": required,
                                "text": text,
                                "time": float(running_time),
                            }
                            fire_sent_for_prompt.clear()
                            fire_time_credit -= 1
                            # environment AP broadcast at the moment we actually create the prompt
                            labeler.advance({"p_firemsg_0_0_0_0"})

                # 2. possible new survivor messages (symptom-based triage)
                if (survivormsg_idx < len(SURVIVORMSG_TIMES)
                        and running_time >= SURVIVORMSG_TIMES[survivormsg_idx]):
                    # Only allow if at least one survivor scan credit exists
                    if surv_credit > 0 and current_surv_prompt is None and len(tasks) > 0:
                        sev = random.choice([1, 2, 3])  # 1=Minor,2=Serious,3=Emergency
                        picks = random.sample(triage_symptoms[sev], k=3)
                        correct = {1:"Minor", 2:"Serious", 3:"Emergency"}[sev]

                        surv_prompt_id += 1
                        current_surv_prompt = {
                            "id": surv_prompt_id,
                            "text": "Assess the patient based on these symptoms:",
                            "symptoms": picks,    # <<< send the 3 symptoms
                            "correct": correct
                        }
                        surv_sent_for_prompt.clear()

                        labeler.advance({"p_survivormsg_0_0_0_0"})
                        surv_credit -= 1
                    # Regardless of whether we used it, move to the next scheduled time
                    survivormsg_idx += 1

                # 3. ATM scheduler (prompt + env broadcast)
                if (atmmsg_idx < len(ATMMSG_TIMES)) and (running_time >= ATMMSG_TIMES[atmmsg_idx]):
                    # If a previous prompt is still pending, mark it failed
                    if current_atm_prompt is not None:
                        atm_results.append(False)
                        # also remove its SEE circle if it existed
                        try:
                            old_id = current_atm_prompt.get("region_id")
                            if old_id:
                                game_mgr.special_regions.remove_region(old_id)
                        except Exception as e:
                            print(f"[ATM] remove previous SEE circle failed: {e}")

                    # Pick coord, build token/text (existing code) ...
                    x, y = _pick_coord_avoiding_targets(ws, grid_size=grid_size)
                    token_tmpl, text_tmpl = random.choice(atm_rows)
                    token = token_tmpl.format(x=x, y=y)
                    text  = text_tmpl.format(x=x, y=y, token=token)

                    atm_prompt_id += 1
                    current_atm_prompt = {
                        "id": atm_prompt_id,
                        "coord": (x, y),
                        "token": token,
                        "text": text,
                        "time": running_time
                    }
                    atm_sent_for_prompt.clear()
                    # Circle
                    try:
                        gui_xy = ws.grid_to_pixel((x, y), grid_size=grid_size, screen_size=(1125, 900))
                        region_id = _atm_region_id(atm_prompt_id)
                        # pass code 3 → SpecialRegions maps to SEE color
                        game_mgr.special_regions.add_region(region_id, gui_xy, 50, 3)
                        current_atm_prompt["region_id"] = region_id
                    except Exception as e:
                        print(f"[ATM] Warning: failed to add SEE circle: {e}")

                    # Broadcast AP & bump index (existing)
                    labeler.advance({"p_atmmsg_0_0_0_0"})
                    atmmsg_idx += 1

                # === Victim detection after p_scan_i ===
                for ap in completed:
                    if ap.startswith("p_scan_") and ap not in victim_detected:
                        for agent in agents_by_type["drones"]:
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
                    if isinstance(ap, str) and ap.startswith("p_verify_") and str(getattr(agent, "role", "")).startswith("humans"):
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

                # === If a human is assigned p_atmconfirm_*, send the ATM prompt now ===
                for agent, ap in assignments.items():
                    if isinstance(ap, str) and ap.startswith("p_atmconfirm_") and str(getattr(agent, "role", "")).startswith("humans"):
                        if current_atm_prompt and (current_atm_prompt["id"] not in atm_sent_for_prompt):
                            buffers['atm_prompt'].append({
                                "id": current_atm_prompt["id"],
                                "text": current_atm_prompt["text"],     # client renders red+bold
                                "token": current_atm_prompt["token"],
                                "coord": list(current_atm_prompt["coord"]),
                            })
                            atm_sent_for_prompt.add(current_atm_prompt["id"])

                # === If a human is assigned p_triage_*, send the survivor prompt now ===
                for agent, ap in assignments.items():
                    if isinstance(ap, str) and ap.startswith("p_triage_") and str(getattr(agent, "role", "")).startswith("humans"):
                        if current_surv_prompt and (current_surv_prompt["id"] not in surv_sent_for_prompt):
                            buffers['surv_prompt'].append({
                                "id": current_surv_prompt["id"],
                                "text": current_surv_prompt["text"],
                                "symptoms": current_surv_prompt.get("symptoms", []),   # <<< include symptoms
                                "choices": ["Emergency", "Serious", "Minor"]
                            })
                            surv_sent_for_prompt.add(current_surv_prompt["id"])

                # === If a human is assigned p_priority_*, send the fire/priority prompt now ===
                for agent, ap in assignments.items():
                    if isinstance(ap, str) and ap.startswith("p_priority_") and str(getattr(agent, "role", "")).startswith("humans"):
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
                        # Remove rejected task
                        tasks = [row for row in tasks if row[0] != idx + 1 ]
                        game_mgr.task = [row for row in game_mgr.task if row[0] != idx + 1 ]
                        buffers['tasks'].append(tasks[:])

                    chosen_gate = labeler.chosen_gate_per_group.get(target_id)
                    if chosen_gate:
                        labeler.advance({verify_ap, chosen_gate})
                    else:
                        labeler.advance({verify_ap})
                    
                    # Award survivor triage credit only after a verification completes
                    surv_credit += 1

                    # bookkeep
                    if image_id is not None:
                        victim_id[idx] = image_id
                    victim_timing[idx] = running_time

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
                    try:
                        game_mgr.special_regions.remove_region(current_fire_prompt["task_id"])
                    except Exception as e:
                        print("[FIRE AUTOCLEAR] remove_region failed:", e)
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
                    else:
                        atm_results.append(False)
                        print(f"[t={running_time:.1f}] ATM confirmation FAIL (expected '{token}', got '{typed}')")

                    # Advance regardless of correctness
                    labeler.advance({"p_atmconfirm_0_3_1_0"})

                    # NEW: remove SEE circle for this prompt
                    try:
                        rid = current_atm_prompt.get("region_id")
                        if rid:
                            game_mgr.special_regions.remove_region(rid)
                    except Exception as e:
                        print(f"[ATM] Warning: failed to remove SEE circle: {e}")

                    # Clear prompt
                    current_atm_prompt = None
                    atm_sent_for_prompt.clear()

            # === Survivor reply handling (top-right buttons) ===
            if isinstance(data, dict) and data.get('surv_reply') is not None:
                payload = data['surv_reply']   # {"id":..., "choice":"Emergency|Serious|Minor"}
                if current_surv_prompt and payload.get("id") == current_surv_prompt["id"]:
                    choice = payload.get("choice")
                    ok = (choice == current_surv_prompt["correct"])
                    surv_results.append(ok)

                    # Advance regardless of correctness
                    labeler.advance({"p_triage_0_3_1_0"})

                    # Clear the prompt, but report correctness in the message
                    current_surv_prompt = None
                    surv_sent_for_prompt.clear()
            
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
            # if time() - wind_time > 3:
            #     wind_time, old_wind_average_speed = update_wind(
            #         game_mgr, wind_time, old_wind_average_speed, n_wind, message)
                
            # === Soket send ===
            # Decide which client to send the message!!!
            if any(len(v) > 0 for v in buffers.values()):
                message_all = {'tasks': None, 'wind_speed': None}
                message_one = {'idx_image': None, 'vic_msg': None, 'workload': None,
                               'atm_prompt': None,
                               'surv_prompt': None,
                               'fire_prompt': None}

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

                if buffers['surv_prompt']:
                    message_one['surv_prompt'] = buffers['surv_prompt'].pop(0)

                if buffers['fire_prompt']:
                    message_one['fire_prompt'] = buffers['fire_prompt'].pop(0)

                try:
                    payload = json.dumps(message_one)  # test serialization
                except TypeError as e:
                    print("[JSON DEBUG] message_one failed:", e)
                    for k, v in message_one.items():
                        try:
                            json.dumps(v)
                        except TypeError as e2:
                            print(f"[JSON DEBUG] key={k}, type={type(v)}, value={v}, err={e2}")
                    raise  # keep the stack trace for now

                # Send messages to clients
                if any(v is not None for v in message_all.values()):
                    for conn, addr in clients:
                        conn.sendall((json.dumps(message_all) + '\n').encode())
                if any(v is not None for v in message_one.values()):
                    # Function allocation between humans
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
                        ws.grid_to_pixel(tuple(p), grid_size=grid_size, screen_size=(1125, 900))
                        for p in agent.path
                    ]
                    # choose a color per type
                    col = (240, 128, 128) if agent.role == 'drones' else (200, 200, 200)
                    game_mgr.paths.append((col, gui_pts))

            # === Draw GUI: Render the game manager ===
            game_mgr.render(vor, centroids)

        # === Drone landing ===
        game_mgr.paths = [] # Clear path
        if landing:
            print(f"[t={running_time:.2f}] Landing drones...")
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
            try:
                conn.sendall((json.dumps({"shutdown": True}) + '\n').encode('utf-8'))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        s.close()
        pygame.quit()
        # Collect data
        print('Clean exit')