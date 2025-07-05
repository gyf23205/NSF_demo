import sys
sys.path.append('D:/Projects/qualisys_drone_sdk')
import pynput
import numpy as np
from time import time, sleep, strftime
from qfly import World
import csv
from rrt_2D import rrt_connect
from gui_panel import GameMgr
from vehicles import VirtualDrone, VirtualGV
import socket
import json


def on_press(key):
    """React to keyboard."""
    global last_key_pressed
    last_key_pressed = key

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def assign_targets_to_drones(starts, targets, landing=None):
    # Initialize paths for each drone
    paths = {i: [starts[i]] for i in range(n_drones)}
    remaining_targets = targets.copy()

    while remaining_targets:
        # Find the nearest target for each drone
        for i in range(n_drones):
            if remaining_targets:
                current_position = paths[i][-1]
                nearest_target = min(remaining_targets, key=lambda p: distance(current_position, p))
                paths[i].append(nearest_target)
                remaining_targets.remove(nearest_target)

    if landing is not None:
        for i in range(len(landing)):
            paths[i].append(landing[i])

    return paths


def assign_targets_with_path_lengths(drones, targets, assigned_drone_index, landing=None):
    # Initialize paths for each drone
    paths = {i: [drones[i]] for i in range(len(drones))}
    remaining_targets = targets.copy()

    # Assign the additional target to the specific drone
    additional_target = remaining_targets.pop(assigned_drone_index)
    paths[assigned_drone_index].append(additional_target)

    while remaining_targets:
        # Find the nearest target for each drone
        for i in range(len(drones)):
            if remaining_targets:
                current_position = paths[i][-1]
                nearest_target = min(remaining_targets, key=lambda p: distance(current_position, p))
                paths[i].append(nearest_target)
                remaining_targets.remove(nearest_target)


if __name__=='__main__':
    ########################## Set up socket ##############################################
    # Create a server for socket communication
    host = '0.0.0.0'  # Use '0.0.0.0' to accept connections from other machines
    port = 8888
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen()
    clients = []  # Track all client addresses
    print("Server waiting for connection...")
    while len(clients) < 1:  # !!! Wait for at least one client to connect
        conn, addr = s.accept()
        print("Connected by", addr)
        clients.append((conn, addr))  # Store the address
    print("All client connected")
    for conn, addr in clients:
        conn.setblocking(False)
    ########################## Set up socket ends ##########################################

    ######################## Environmental setting ###################################
    # Set operation mode
    fa = 1  # {0:Automatic allocation; 1: human supervision}

    # Constants
    n_drones = 4
    n_gvs = 2
    n_targets = 7
    speed_constant = 5.0
    stay_duration_drone = 5
    hover_duration = 10
    stay_distance_drone = 0.1
    n_wind = 3

    # Define world
    world = World()

    # Listen to the keyboard
    listener = pynput.keyboard.Listener(on_press=on_press)
    listener.start()

    # Time
    t = time()
    dt = 0
    wind_time = 0
    dt_prev = [hover_duration for _ in range(n_drones)]

    # Create drones and GVs
    drones = [VirtualDrone(i, tuple(2 * np.random.random(2,) - 1)) for i in range(n_drones)]
    gvs = [VirtualGV(i, tuple(2 * np.random.random(2,) - 1)) for i in range(n_gvs)]
    takeoff_positions = [d.position[0:2] for d in drones]

    # Define flags and time limits
    fly = True
    last_key_pressed = None
    mission_complete = False
    message_changed = True
    stay_flag_drone = [True for _ in range(n_drones)]
    stay_time_drone = [0 for _ in range(n_drones)]
    start_time_drone = [0 for _ in range(n_drones)]

    # Randomly generate targets
    random_positions = []
    while len(random_positions) < n_targets:
        new_position = [np.random.uniform(-1.5, 1.5), np.random.uniform(-1, 1)]

        # Check distance to existing positions
        if all(distance(new_position, existing) >= 0.55 for existing in random_positions + takeoff_positions):
            # Avoid takeoff and wind position by force
            if distance(new_position, [0, 0]) > 0.7:
                random_positions.append(new_position)
    # Known target and new target
    new_target = random_positions[-1]
    target_positions = random_positions[0:n_targets - 1]
    target_index_drone = [0 for _ in range(n_drones)]
    target_distance_drone = [0 for _ in range(n_drones)]

    # Path
    path_index_drone = [0 for _ in range(n_drones)]
    path_index_gv = [0 for _ in range(n_gvs)]

    # Survivor images
    survivor_images = set()
    while len(survivor_images) < n_targets:
        random_number = np.random.randint(low=1, high=21)
        survivor_images.add(random_number)
    survivor_images = list(survivor_images)
    survivor_index = 0

    # Task assignment
    drone_paths = assign_targets_to_drones(takeoff_positions, target_positions, landing=takeoff_positions)
    
    # Wind
    old_wind_average_speed = 0.0

    # New target
    update_on = np.random.normal(loc=18.0, scale=1.0)

    # RRT-connect for all drone paths
    drone_trajectory = [[] for _ in range(n_drones)]
    for d_inx in range(n_drones):
        for p_idx in range(len(drone_paths[d_inx]) - 1):
            rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx], drone_paths[d_inx][p_idx + 1], 0.08, 0.05, 5000)
            rrt_conn.planning()
            rrt_conn.smoothing()
            drone_trajectory[d_inx].append(rrt_conn.path)

    game_mgr = GameMgr(drones, gvs)
    target_gui = game_mgr.position_meter_to_gui(target_positions)
    takeoff_gui = game_mgr.position_meter_to_gui(takeoff_positions)
    # Remaining paths for re-planning
    target_remaining = target_positions.copy()
    tasks = []
    for idx, t_gui in enumerate(target_gui):
        tasks.append([idx + 1, t_gui.tolist(), 0])
    game_mgr.set_target(target=target_gui)
    game_mgr.set_takeoff_positions(takeoff_gui)
    game_mgr.set_task(tasks)
    # Send initial tasks to the clients
    message = {'idx_image': None, 'tasks': tasks, 'wind_speed': None, 'progress': None, 'workload': None} # Message to be sent to the clients
    for idx in range(len(clients)):
        clients[idx][0].sendall(json.dumps(message).encode())
    game_mgr.render()
    ######################## Environmental setting ends ###################################

    ########################## Main loop ################################################
    while fly:
        data = None # Data received from the clients
        message = {'idx_image': None, 'tasks': None, 'wind_speed': None, 'progress': None, 'workload': None} # Message to be sent to the clients

        # Land with ESC
        if last_key_pressed == pynput.keyboard.Key.esc:
            print(f'[t={int(dt)}] Escape by keyboard.')
            break

        dt = time() - t

        if mission_complete: # Mission_flag should be given by the function allocation module
            print(f'[t={int(dt)}] All missions completed!')
            break

        if len(tasks) > 0:
            mission_complete = False

        ############################ Socket receive ########################### !!!
        try:
            data = clients[0][0].recv(1024).decode() # !!! Decide which client to listen
            if data:
                data = json.loads(data)
        except BlockingIOError:
            pass
        ############################# Socket receive ends ###########################
        
        ########################### Update human progress and workload ##############
        ########################### Update human progress and workload ends #########

        ############################ Task update ################################
        if data and data['tasks'] is not None:
            exist_task_idx = []
            for task in data['tasks']:
                exist_task_idx.append(task['task_id'])
                for j, ta in enumerate(tasks):
                    if ta[0] == task['task_id']:
                        tasks[j][2] = task['priority']
                        print(f'reset task {task["task_id"]} priority to {task["priority"]}')
                        break
                
            for j, ta in enumerate(tasks):
                if ta[0] not in exist_task_idx:
                    tasks.pop(j)
                    print(f'[t={int(dt)}] Task {ta[0]} removed from the task list.')
        ############################## Task update ends ##############################

        ########################### Drone loop #################################
        # Hover at first
        if dt < hover_duration:
            for d in game_mgr.drones:
                d.takeoff_in_place(0.5*world.expanse[2])
            sleep(0.01)
        else:
            altitude_adjust = [0 for _ in range(n_drones)]
            for idx, d in enumerate(drones):
                target_current = drone_paths[idx][target_index_drone[idx] + 1]
                # Temporary
                current_trajectory = drone_trajectory[idx][target_index_drone[idx]]
                # Find the target positions: {make indexing as a function for better interpretability}
                increment = speed_constant * (dt - dt_prev[idx])
                path_index_drone[idx] += increment
                picked = min(int(path_index_drone[idx]) + 1, len(current_trajectory))
                target = [current_trajectory[-picked][0], current_trajectory[-picked][1],
                        0.5 * world.expanse[2] + altitude_adjust[idx]]
                drones[idx].set_position(target)
                
                # Time
                dt_prev[idx] = dt
                sleep(0.01)

                # Health
                drones[idx].health -= 0.01

                # Check distance to the target
                target_distance_drone[idx] = distance(target_current, drones[idx].position[0:2])
                # Check stay start time
                if target_distance_drone[idx] < stay_distance_drone:
                    # Measure stay time
                    if stay_flag_drone[idx]:
                        # Measure: start point
                        start_time_drone[idx] = time()
                        stay_flag_drone[idx] = False
                    # Measure: end point
                    stay_time_drone[idx] = time() - start_time_drone[idx]

                # Check stay duration
                if stay_time_drone[idx] > stay_duration_drone:
                    if target_index_drone[idx] < len(drone_trajectory[idx]) - 1:
                        # Generate victim
                        if not game_mgr.victim_detected[idx]:
                            game_mgr.victim_id[idx] = survivor_images[survivor_index]
                            # Send the victim idx to the according user
                            message['idx_image'] = str(survivor_images[survivor_index])
                            message_changed = True
                            survivor_index += 1
                            game_mgr.victim_detected[idx] = True
                            # To record response time
                            game_mgr.victim_timing[idx] = dt

                        # Once victim is selected, close it: only if there is no unassigned target
                        if game_mgr.target_decided:
                            if data and data['victim'] is not None:
                                game_mgr.victim_clicked[idx] = 1 if data['victim'] == 'accept' else 2
                                data['victim'] = None
        
                            if game_mgr.victim_clicked[idx]:
                                game_mgr.victim_id[idx] = 0
                                game_mgr.victim_detected[idx] = False
                                # To record response time
                                # game_mgr.missions[idx].response_time.append(dt - game_mgr.victim_timing[idx])
                                # game_mgr.missions[idx].time_stamp.append(dt)
                                # game_mgr.missions[idx].drone_id.append(idx)
                                # game_mgr.victim_timing[idx] = 0

                                # print(f'Response time by drone {int(idx + 1)}: {game_mgr.missions[idx].response_time[-1]:.2f} sec')
                                # Print status
                                print(f'[t={int(dt)}] Drone {int(idx + 1)}: target {int(target_index_drone[idx] + 1)} accomplished')
                                # Remove the task whose target matches target_current
                                for i, task in enumerate(tasks):
                                    if np.allclose(task[1], target_current):
                                        tasks.pop(i)
                                        message['tasks'] = tasks
                                        print('task updated in message')
                                        message_changed = True
                                        break
                                # target_remaining.remove(target_current)
                                # Update and reset
                                target_index_drone[idx] += 1
                                stay_flag_drone[idx] = True
                                stay_time_drone[idx] = 0
                                # Temporary
                                path_index_drone[idx] = 0
                        else:
                            game_mgr.victim_block_choice[idx] = True
        ########################### Drone loop ends ############################

        ############################ GV loop ###################################
        # !!!
        ############################ GV loop ends ##############################
        
        ########################### Awareness map #####################################
        pos = [(d.position[0], d.position[1]) for d in drones]
        drone_gui_positions = game_mgr.position_meter_to_gui(pos)
        game_mgr.update_awareness(drone_gui_positions)
        ########################### Awareness map ends #################################

        ####################### Wind condition #####################################
        # For every 5 seconds, sample n_wind number of wind positions
        # Wind is represented as (x, y, radius, speed)
        if time() - wind_time > 5:
            wind_time = time()
            wind = [[np.random.uniform(-1.9, 1.9), np.random.uniform(-1.25, 1.25), np.random.uniform(0.05, 0.15)] for _ in range(n_wind)]
            game_mgr.wind = []
            for w in wind:
                game_mgr.set_wind(w)
            average_wind_speed = np.mean([w[2] for w in wind]) if wind else 0.0
            # Update the wind speed in the message if it has changed significantly
            if abs(average_wind_speed - old_wind_average_speed) > 1:
                # Current way of changing wind speed is not very significant, can change it.
                message['wind_speed'] = average_wind_speed
                message_changed = True
                old_wind_average_speed = average_wind_speed
        
        if data and data['weather_decision'] is not None:
            if data['weather_decision'] == 'change':
                print(f'[t={int(dt)}] Weather condition changed by user.')
                data['weather_decision'] = None
                current_start = [[drones[idx].position[0], drones[idx].position[1]] for idx in range(n_drones)]
                drone_paths = assign_targets_to_drones(current_start, target_remaining,
                                                    landing=takeoff_positions)
                # Treated as a obstacles in RRT
                obs = [w[0:3] for w in wind]
                # RRT-connect for all drone paths
                drone_trajectory = [[] for _ in range(n_drones)]
                for d_inx in range(n_drones):
                    for p_idx in range(len(drone_paths[d_inx]) - 1):
                        rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx],
                                                        drone_paths[d_inx][p_idx + 1], 0.08, 0.05, 5000)
                        rrt_conn.utils.update_obs(obs, [], [])
                        rrt_conn.planning()
                        rrt_conn.smoothing()
                        drone_trajectory[d_inx].append(rrt_conn.path)
                target_index = [0 for _ in range(n_drones)]
                # Time correction
                path_index = [0 for _ in range(n_drones)]
                dt_prev = [dt for _ in range(n_drones)]
            elif data['weather_decision'] == 'maintain':
                print(f'[t={int(dt)}] Weather condition maintained by user.')
                data['weather_decision'] = None
                speed_constant = 5.0
        ################## Wind condition ends #####################################

        ########################### Socket Send #####################################
        # Decide which client to send the message!!!
        if message_changed:
            idx = 0
            # print('Message changed')
            # print(message)
            clients[idx][0].sendall(json.dumps(message).encode())
            message_changed = False
            # assert False
        ############################# Socket Send ends #####################################
        game_mgr.render()
        # print('gui rendered')

    # Land till the end
    while max([drones[idx].position[2] for idx in range(2)]) > 0.01:
        for idx in range(n_drones):
            drones[idx].land_in_place()
            # sleep(0.01)
            drones[idx].position = game_mgr.position_meter_to_gui([drones[idx].position])
            drones[idx].rt = np.random.normal(0, 0.01, 1)[0]
        game_mgr.render()

    # Close the socket
    for conn, addr in clients:
        conn.close()
    
    # Close the listener
    listener.stop()

    # Collect data
    print('Collecting data...')