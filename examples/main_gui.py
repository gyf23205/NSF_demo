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
import pygame
from scipy.spatial import Voronoi
from skimage import measure
import matplotlib.pyplot as plt
from constants import IMAGE_PATH
from PIL import Image


def on_press(key):
    """React to keyboard."""
    global last_key_pressed
    last_key_pressed = key

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def assign_targets_to_drones(starts, targets, landing=None):
    # Initialize paths for each drone
    # !!! Need a new way to assign targets to drones.
    # Now all the drones take off at the same place and all the targets are assigned to the first drones.
    paths = {i: [starts[i]] for i in range(n_drones)}
    remaining_targets = targets.copy()

    while remaining_targets:
        # Find the nearest target for each drone
        for i in range(n_drones):
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

def generate_voronoi_plots(map, centroids):
        # Pad map to (1200, 960)
        padded_map = np.zeros((960, 1200))
        padded_map[:map.shape[0], :map.shape[1]] = map
        map = padded_map
        centroids = np.array(centroids)
        # Compute power diagram
        label_map = np.zeros(map.shape, dtype=int)
        Y, X = np.meshgrid(np.arange(map.shape[0]), np.arange(map.shape[1]), indexing='ij')
        # points = np.stack((X, Y), axis=-1)  # shape (H, W, 2)

        for i, (px, py) in enumerate(centroids):
            dx = X - px
            dy = Y - py
            dist2 = dx**2 + dy**2 - map[i]**2
            if i == 0:
                dist_stack = dist2[..., np.newaxis]
            else:
                dist_stack = np.concatenate((dist_stack, dist2[..., np.newaxis]), axis=2)

        label_map = np.argmin(dist_stack, axis=2)

        region_vertices = {}  # key: region index, value: list of (y, x) coords

        for i in range(len(centroids)):  # N is the number of sites
            mask = (label_map == i).astype(np.uint8)
            contours = measure.find_contours(mask, 0.5)
            
            if contours:
                # You can get multiple disconnected contours — take the largest
                largest = max(contours, key=lambda x: x.shape[0])
                region_vertices[i] = largest 
        
        fig_width, fig_height = 1200, 960
        dpi = 100
        plt.xlim(0, fig_width)
        plt.ylim(fig_height, 0)  # Invert y for image coordinates if needed
        plt.figure(figsize=(fig_width/dpi, fig_height/dpi), dpi=dpi)
        plt.axis('off')
        # plt.scatter(target_gui[:,0], target_gui[:,1], c='red', edgecolors='black', s=80, label='Sites')
        for i, contour in region_vertices.items():
            plt.plot(contour[:,1], contour[:,0], linewidth=1.5, label=f'Region {i}')

        plt.savefig('examples/images/voronoi_regions.png', pad_inches=0, dpi=dpi, transparent=False)
        plt.close()
        img = Image.open('examples/images/voronoi_regions.png')
        box = (150, 120, 1200-150, 960-120)
        cropped_img = img.crop(box)
        cropped_img = cropped_img.transpose(Image.FLIP_TOP_BOTTOM)
        cropped_img.save('examples/images/voronoi_regions_cropped.png', format='PNG')

if __name__=='__main__':
    try:
        ########################## Set up socket ##############################################
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
        ########################## Set up socket ends ##########################################

        ######################## Environmental setting ###################################
        # Set operation mode
        fa = 1  # {0:Automatic allocation; 1: human supervision}

        # Constants
        n_drones = 2
        n_gvs = 2
        n_targets = 6
        speed_constant = 2.0  # Speed of the drones
        stay_duration_drone = 8
        hover_duration = 10
        stay_distance_drone = 0.1
        n_wind = 3
        cruzing_altitude = 0.7

        # Define world
        world = World()

        # Listen to the keyboard
        listener = pynput.keyboard.Listener(on_press=on_press)
        listener.start()
        print('Keyboard listener started')
        # Time
        t = time()
        dt = 0
        wind_time = 0
        dt_prev = [hover_duration for _ in range(n_drones)]
        print('Time initialized')
        # Create drones and GVs
        # drones = [VirtualDrone(i, (-1.55, -1.3)) for i in range(n_drones)]
        # gvs = [VirtualGV(i, (-1.55, -1.3)) for i in range(n_gvs)]
        drones = [VirtualDrone(i, tuple(2 * np.random.random(2,) - 1)) for i in range(n_drones)]
        gvs = [VirtualGV(i, tuple(2 * np.random.random(2,) - 1)) for i in range(n_gvs)]
        # print(len(gvs))
        # assert False
        takeoff_positions = [d.position[0:2] for d in drones]
        print('Drones and GVs created')
        # Define flags and time limits
        fly = True
        last_key_pressed = None
        # mission_complete = False
        message_changed = True
        stay_flag_drone = [True for _ in range(n_drones)]
        stay_time_drone = [0 for _ in range(n_drones)]
        start_time_drone = [0 for _ in range(n_drones)]
        print('Flags and time limits initialized')
        # Randomly generate targets
        random_positions = []
        while len(random_positions) < n_targets:
            # print('Generating random target positions')
            new_position = [np.random.uniform(-1.9, 1.9), np.random.uniform(-1.2, 1.2)]

            # Check distance to existing positions
            if all(distance(new_position, existing) >= 0.25 for existing in random_positions + takeoff_positions):
                # Avoid takeoff and wind position by force
                if distance(new_position, [0, 0]) > 0.5:
                    random_positions.append(new_position)

        # Generate Voronoi targets
        voronoi_seeds = np.array(random_positions)
        # Compute Voronoi diagram
        vor = Voronoi(voronoi_seeds)

        # Compute centroids of finite Voronoi regions
        def region_centroid(region, vertices):
            pts = np.array([vertices[i] for i in region])
            return np.mean(pts, axis=0)

        centroids = []
        for i, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if not region or -1 in region:
                # Infinite region: use the seed point itself
                centroids.append(vor.points[i].tolist())
            else:
                # Finite region: use the centroid of the polygon
                centroid = region_centroid(region, vor.vertices)
                centroids.append(centroid.tolist())

        # Known target and new target
        # new_target = random_positions[-1]
        target_positions = random_positions
        target_index_drone = [0 for _ in range(n_drones)]
        target_distance_drone = [0 for _ in range(n_drones)]
        print('Target positions generated')
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
        assignment = {}
        for idx, positions in drone_paths.items():
            for pos in positions:
                assignment[str(pos)] = idx + 1  # Assign drone index starting from 1
        # print(assignment)
        # assert False
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
        # print(len(gvs))
        # assert False
        game_mgr = GameMgr(drones, gvs)
        target_gui = game_mgr.position_meter_to_gui(target_positions)
        takeoff_gui = game_mgr.position_meter_to_gui(takeoff_positions)
        # Remaining paths for re-planning
        target_remaining = target_positions.copy()
        tasks = []
        # mission_complete = [False for _ in range(n_drones)]
        for idx, (t_gui, t_pos) in enumerate(zip(target_gui, target_positions)):
            tasks.append([idx + 1, t_gui.tolist(), 0, assignment[str(t_pos)]])
        game_mgr.set_target(target=target_gui)
        game_mgr.set_takeoff_positions(takeoff_gui)
        game_mgr.set_task(tasks)
        # Send initial tasks to the clients
        message = {'idx_image': None, 'tasks': tasks, 'wind_speed': None, 'progress': None, 'workload': None, 'vic_msg': None} # Message to be sent to the clients
        for idx in range(len(clients)):
            clients[idx][0].sendall(json.dumps(message).encode())
        print('Init finished')
        game_mgr.render(vor, random_positions)
        print('GUI rendered')
        ######################## Environmental setting ends ###################################

        ########################## Main loop ################################################
        centers = [None for _ in range(n_drones)]
        just_taken_off = [True for _ in range(n_drones)]
        while fly:
            sleep(0.01)  # To avoid high CPU usage
            # print('flying')
            data = None # Data received from the clients
            message = {'idx_image': None, 'tasks': None, 'wind_speed': None, 'progress': None, 'workload': None, 'vic_msg': None} # Message to be sent to the clients

            # Land with ESC
            if last_key_pressed == pynput.keyboard.Key.esc:
                print(f'[t={int(dt)}] Escape by keyboard.')
                fly = False

            dt = time() - t

            # if mission_complete: # Mission_flag should be given by the function allocation module
            #     print(f'[t={int(dt)}] All missions completed!')
            #     break

            if not tasks:  # If no tasks left and all the drones are at the takeoff position
                print(f'[t={int(dt)}] No tasks left, all drones at takeoff position.')
                if all(np.allclose(d.position[0:2], takeoff_positions[i]) for i, d in enumerate(drones)):
                    print(f'[t={int(dt)}] All drones at takeoff position.')
                    fly = False

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
                        if ta[0] == task['task_id'] and ta[2] != task['priority']:
                            ta[2] = task['priority']
                            print(f'reset task {task["task_id"]} priority to {task["priority"]}')
                            break
                    
                # for j, ta in enumerate(tasks):
                #     if ta[0] not in exist_task_idx:
                #         tasks.pop(j)
                #         print(f'[t={int(dt)}] Task {ta[0]} removed from the task list.')
            ############################## Task update ends ##############################

            ########################### Drone loop #################################
            # Hover at first
            if dt < hover_duration:
                for d in game_mgr.drones:
                    d.takeoff_in_place(cruzing_altitude * world.expanse[2])
                    d.status = 'taking off'
                    # print(dt)
                sleep(0.01)
            else:
                # altitude_adjust = [0 for _ in range(n_drones)]
                for idx, d in enumerate(drones):
                    if just_taken_off[idx]:
                        d.status = 'flying'
                        just_taken_off[idx] = False
                    target_current = drone_paths[idx][target_index_drone[idx] + 1]
                    # Temporary
                    current_trajectory = drone_trajectory[idx][target_index_drone[idx]]
                    # Find the target positions: {make indexing as a function for better interpretability}
                    increment = speed_constant * (dt - dt_prev[idx])
                    path_index_drone[idx] += increment
                    picked = min(int(path_index_drone[idx]) + 1, len(current_trajectory))
                    target = [current_trajectory[-picked][0], current_trajectory[-picked][1],
                            d.position[2]]  # Keep the altitude same as the drone's current position
                    drones[idx].set_position(target)

                    # if target_index_drone[idx] == len(drone_paths[idx]) - 1:
                    #     mission_complete[idx] = True

                    # Time
                    dt_prev[idx] = dt
                    sleep(0.01)

                    # Health
                    drones[idx].health -= 0.01

                    # Check distance to the target
                    target_distance_drone[idx] = distance(target_current, drones[idx].position[0:2])
                    # Check stay start time
                    if target_distance_drone[idx] < stay_distance_drone:
                        if target_index_drone[idx] < len(drone_trajectory[idx]) - 1:
                            # Measure stay time
                            if stay_flag_drone[idx]:
                                # Measure: start point
                                start_time_drone[idx] = time()
                                stay_flag_drone[idx] = False
                                # Save the current trajectory to restore later
                                if not hasattr(d, 'saved_trajectory'):
                                    d.saved_trajectory = current_trajectory
                                # Generate a circular trajectory around the target
                                centers[idx] = np.array(target_current)
                                radius = 0.3  # Circle radius in meters
                                num_points = 80
                                theta = np.linspace(0, 2 * np.pi, num_points)
                                circle_traj = [
                                    [centers[idx][0] + radius * np.cos(t), centers[idx] [1] + radius * np.sin(t), d.position[2]]
                                    for t in theta
                                ]
                                d.circle_trajectory = circle_traj
                                d.circle_index = 0

                            # Follow the circular trajectory
                            if hasattr(d, 'circle_trajectory'):
                                d.set_position(d.circle_trajectory[d.circle_index])
                                d.circle_index = (d.circle_index + 1) % len(d.circle_trajectory)
                        # Measure: end point
                        stay_time_drone[idx] = time() - start_time_drone[idx]
                        # d.down4inspect()
                        d.status = 'inspecting'

                    # Adjust altitude given status
                    if d.status == 'inspecting':
                        d.down4inspect()
                    elif d.status == 'flying':
                        d.up4move()

                    # Check stay duration
                    if stay_time_drone[idx] > stay_duration_drone:
                        if target_index_drone[idx] < len(drone_trajectory[idx]) - 1:
                            # Generate victim
                            if not game_mgr.victim_detected[idx]:
                                game_mgr.victim_id[idx] = survivor_images[survivor_index]
                                # Send the victim idx to the according user
                                message['idx_image'] = str(survivor_images[survivor_index])
                                need_help = np.random.choice([True, False], p=[0.5, 0.5])
                                if need_help:
                                    pos_rounded = [round(coord, 2) for coord in d.position[0:2]]
                                    message['vic_msg'] = f'Drone {idx + 1} receive a survivor message at {pos_rounded}, please response!'
                                message_changed = True
                                survivor_index += 1
                                game_mgr.victim_detected[idx] = True
                                # To record response time
                                game_mgr.victim_timing[idx] = dt

                            # Once victim is selected, close it: only if there is no unassigned target
                            # if game_mgr.target_decided:
                            if data and data['victim'] is not None:
                                if data['victim'] == 'accept':
                                    game_mgr.victim_clicked[idx] = 1
                                elif data['victim'] == 'reject':
                                    game_mgr.victim_clicked[idx] = 2
                                elif data['victim'] == 'handover':
                                    # !!! Need hand over logic
                                    game_mgr.victim_clicked[idx] = 3
                                data['victim'] = None
        
                            if game_mgr.victim_clicked[idx]:
                                # Reset the clicked status
                                game_mgr.victim_clicked[idx] = 0
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
                                # Remove the task whose target matches center
                                for i, task in enumerate(tasks):
                                    if np.allclose(task[1], game_mgr.position_meter_to_gui([centers[idx]])[0], atol=1):
                                        centers[idx] = None  # Reset the center
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
                                stay_time_drone[idx] = 0
                                # Restore previous trajectory if needed
                                if hasattr(d, 'saved_trajectory'):
                                    current_trajectory = d.saved_trajectory
                                    del d.saved_trajectory
                                if hasattr(d, 'circle_trajectory'):
                                    del d.circle_trajectory
                                    del d.circle_index
                                # d.up4move()
                                d.status = 'flying'
                            else:
                                game_mgr.victim_block_choice[idx] = True
            ########################### Drone loop ends ############################

            ############################ GV loop ###################################
            # !!!
            ############################ GV loop ends ##############################
            
            ########################### Awareness map #####################################
            pos = [(d.position[0], d.position[1]) for d in drones]
            drone_gui_positions = game_mgr.position_meter_to_gui(pos)
            game_mgr.update_awareness(drone_gui_positions, radius=40)
            ########################### Awareness map ends #################################

            ####################### Wind condition #####################################
            # For every 5 seconds, sample n_wind number of wind positions
            # Wind is represented as (x, y, radius, speed)
            if time() - wind_time > 3:
                wind_time = time()
                if not game_mgr.wind:
                    wind = [[np.random.uniform(-1.9, 1.9), np.random.uniform(-1.25, 1.25), np.random.uniform(0.1, 0.25)] for _ in range(n_wind)]
                else:
                    wind = game_mgr.wind.copy()
                    increment = [[np.random.uniform(-0.1, 0.1), np.random.uniform(-0.1, 0.1), np.random.uniform(-0.03, 0.03)] for _ in range(n_wind)]
                    for i in range(n_wind):
                        wind[i][0] += increment[i][0]
                        wind[i][1] += increment[i][1]
                        wind[i][2] += increment[i][2]
                        # Ensure the wind position is within the world bounds
                        wind[i][0] = np.clip(wind[i][0], -1.9, 1.9)
                        wind[i][1] = np.clip(wind[i][1], -1.25, 1.25)
                        wind[i][2] = np.clip(wind[i][2], 0.05, 0.15)
                game_mgr.reset_wind()
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
                elif data['weather_decision'] == 'handover':
                    # !!! Need hand over logic
                    print(f'[t={int(dt)}] Weather condition handover by user.')
                    data['weather_decision'] = None
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
            game_mgr.render(vor, random_positions)
            # print('gui rendered')

        # Land till the end
        print('Landing...')
        while max([drones[idx].position[2] for idx in range(n_drones)]) > 0.01:
            for idx in range(n_drones):
                drones[idx].land_in_place()
                print(f'Drone {idx + 1} landing, altitude: {drones[idx].position[2]:.2f}')
                drones[idx].status = 'landing'
                sleep(0.01)
                # drones[idx].position = game_mgr.position_meter_to_gui([drones[idx].position])
                drones[idx].rt = np.random.normal(0, 0.01, 1)[0]
            game_mgr.render(vor, random_positions)

        clicked = False
        while not clicked:
            for event in pygame.event.get():
                clicked = game_mgr.workload_render(event)

        clicked = False
        while not clicked:
            for event in pygame.event.get():
                clicked = game_mgr.perceived_risk_render(event)
    finally:
        # Close the socket
        for conn, addr in clients:
            conn.close()
        s.close()
        # Close the listener
        listener.stop()
        pygame.quit()
        # Collect data
        print('Clean exit')