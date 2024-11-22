"""
qfly | Qualisys Drone SDK based Code
https://github.com/qualisys/qualisys_drone_sdk

Human-Autonomy Teaming Testbed
Sooyung Byeon, Purdue University
November, 2024

ESC to land at any time.
"""

import pynput
import numpy as np
from time import sleep, time, strftime
from qfly import Pose, QualisysCrazyflie, World, ParallelContexts, utils
import csv
from rrt_2D import rrt_connect
import game

import math
import threading

# Additional import for workload estimation
import os
import glob
import sys
import shutil

# Participant ID
participant = 'p11' + 'h'

########################################################################################################################
# Allow importing bleakheart from parent directory
sys.path.append('../')
Openface_directory = "C:/Users/sooyung/OneDrive - purdue.edu/Desktop/Repository/data/Openface/"  # Openface output directory
ECG_directory = "C:/Users/sooyung/OneDrive - purdue.edu/Desktop/Repository/data/ECG/"
try:
    latest_csv = max(glob.glob(Openface_directory + '*.csv'), key=os.path.getctime)  # give path to your desired file path
    print(latest_csv)
except ValueError:
    print('No csv file at Openface.')
########################################################################################################################

# Drone setups
# QTM rigid body names
cf_body_names = [
    'nsf1',
    'nsf11'
]
# Crazyflie addresses
cf_uris = [
    'radio://0/80/2M/E7E7E7E711',
    'radio://0/80/2M/E7E7E7E731'
]
# Crazyflie marker ids
cf_marker_ids = [
    [1, 2, 3, 4],
    [11, 12, 13, 14]
]

# Function allocation
fa = 3  # {1: monitor + confirm, 2: + re-planning, 3: + fault, verbal reporting}

# Result dataset directory
dataset_directory = "C:/Users/sooyung/OneDrive - purdue.edu/Desktop/Repository/data/Dataset/FA{}/".format(fa)

# Drone Setting: Physical constraints
hover_duration = 10
stay_duration = 5
stay_distance = 0.1
separation_distance = 0.5
total_duration = 180
speed_constant = 3.0

# Set up world - the World object comes with sane defaults (geofencing)
world = World()

# Watch key presses with a global variable
last_key_pressed = None

# Log Setting: File name and directory
time_string = strftime('%y%m%d%H%M%S')
log_name = '../logs/log' + time_string + '_' + participant + '.csv'

# Open logging csv file: put head (variable names)
with open(log_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['time', 'pos_x1', 'pos_y1', 'pos_z1', 'yaw1', 'pos_x2', 'pos_y2', 'pos_z2', 'yaw2'])


def log_function(qcfs_):
    # Compute yaw angles
    yaw1 = -math.atan2(qcfs_[0].pose.rotmatrix[1][0], qcfs_[0].pose.rotmatrix[0][0])
    yaw2 = -math.atan2(qcfs_[1].pose.rotmatrix[1][0], qcfs_[1].pose.rotmatrix[0][0])
    with open(log_name, 'a', newline='') as file_:
        writer_line = csv.writer(file_)
        writer_line.writerow(
            [time(), qcfs_[0].pose.x, qcfs_[0].pose.y, qcfs_[0].pose.z, yaw1,
             qcfs_[1].pose.x, qcfs_[1].pose.y, qcfs_[1].pose.z, yaw2])


def call_log_function_period(period, stop, *args):
    if not stop.is_set():
        log_function(*args)
        threading.Timer(period, call_log_function_period, [period, stop] + list(args)).start()


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def assign_targets_to_drones(drones, targets, landing=None):
    # Initialize paths for each drone
    paths = {i: [drones[i]] for i in range(len(drones))}
    remaining_targets = targets.copy()

    while remaining_targets:
        # Find the nearest target for each drone
        for i in range(len(drones)):
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

    # Add landing positions if provided
    if landing is not None:
        for i in range(len(landing)):
            paths[i].append(landing[i])

    # Compute path lengths for each drone
    path_lengths = []
    for i in range(len(drones)):
        path_length = sum(distance(paths[i][j], paths[i][j + 1]) for j in range(len(paths[i]) - 1))
        path_lengths.append(path_length)

    # Output 0 is total path length (output[0] = output[1] + output[2])
    output = [sum(path_lengths), path_lengths[0], path_lengths[1]]

    return output, paths


# Current version cannot be extended for 3+ drones cases
def avoid_altitude_control(drones, proximity=separation_distance):
    altitude = [0, 0]
    # If they are too close, ...
    if distance(drones[0], drones[1]) < proximity:
        altitude[0] = 0.3
        altitude[1] = -0.3

    return altitude


# Set up keyboard callback
def on_press(key):
    """React to keyboard."""
    global last_key_pressed
    last_key_pressed = key


# Listen to the keyboard
listener = pynput.keyboard.Listener(on_press=on_press)
listener.start()

# Stack up context managers
_qcfs = [QualisysCrazyflie(cf_body_name,
                           cf_uri,
                           world,
                           marker_ids=cf_marker_id,
                           qtm_ip="192.168.123.2")
         for cf_body_name, cf_uri, cf_marker_id
         in zip(cf_body_names, cf_uris, cf_marker_ids)]

with ParallelContexts(*_qcfs) as qcfs:
    print("Beginning maneuvers...")

    # "fly" variable used for landing on demand
    fly = True

    # Log function call
    stop_event = threading.Event()
    call_log_function_period(0.1, stop_event, qcfs)

    ####################################################################################################################
    # Task allocation: mission and computation
    takeoff_positions = [[qcfs[0].pose.x, qcfs[0].pose.y], [qcfs[1].pose.x, qcfs[1].pose.y]]
    random_positions = []
    num_targets = 7
    while len(random_positions) < num_targets:
        new_position = [np.random.uniform(-2.3, 2.3), np.random.uniform(-1.2, 1.2)]

        # Check distance to existing positions
        if all(distance(new_position, existing) >= 0.55 for existing in random_positions + takeoff_positions):
            # Avoid takeoff and wind position by force
            if distance(new_position, [0, 0]) > 0.7:
                random_positions.append(new_position)

    # Known target and new target
    new_target = random_positions[-1]
    target_positions = random_positions[0:num_targets - 1]

    # Survivor images
    survivor_images = set()
    while len(survivor_images) < num_targets:
        random_number = np.random.randint(low=1, high=21)
        survivor_images.add(random_number)
    survivor_images = list(survivor_images)
    survivor_index = 0

    # Task assignment in advance
    drone_paths = assign_targets_to_drones(takeoff_positions, target_positions)

    # Wind position and timing
    wind = [0, 0, 0.5]
    wind_on = np.random.normal(loc=30.0, scale=3.0)
    wind_off = np.random.normal(loc=70.0, scale=3.0)

    # Report timing
    report_on = np.random.normal(loc=55.0, scale=3.0)

    # New target
    update_on = np.random.normal(loc=18.0, scale=1.0)

    # Remaining paths for re-planning
    target_remaining = target_positions.copy()
    path1 = []
    path2 = []

    # RRT connect for all drone paths (targets)
    drone_trajectory = [[], []]
    for d_inx in range(2):
        for p_idx in range(len(drone_paths[d_inx]) - 1):
            rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx], drone_paths[d_inx][p_idx + 1], 0.08, 0.05, 5000)
            rrt_conn.planning()
            rrt_conn.smoothing()
            drone_trajectory[d_inx].append(rrt_conn.path)
    ####################################################################################################################

    # Initialize
    target_index = [0, 0]
    target_distance = [0, 0]

    # Stay flag
    stay_flag = [True, True]
    stay_time = [0, 0]
    start_time = [0, 0]

    # Mission completion
    mission_complete = [False, False]

    # Load game environment
    game_mgr = game.GameMgr()
    game_mgr.update()

    # Target and takeoff positions for GUI
    target_gui = game_mgr.position_meter_to_gui(target_positions)
    takeoff_gui = game_mgr.position_meter_to_gui(takeoff_positions)
    game_mgr.set_target(target=target_gui)
    game_mgr.set_takeoff_positions(takeoff_gui)

    # Time
    t = time()
    dt = 0
    dt_prev = [hover_duration, hover_duration]

    # Path
    path_index = [0, 0]

    ####################################################################################################################
    # Erase current log (HR, Openface)
    # 1. Heart rate
    frame_old = []
    error_ind = False
    ecg_csv = ECG_directory + 'ecg.csv'
    try:
        with open(str(ecg_csv), 'w') as my_file:
            wr = csv.writer(my_file)
            wr.writerow(frame_old)
    except:
        error_ind = True
        print('Error detected: HR')

    # 2. Openface feature receiver: 429 rows vector
    try:
        with open(str(latest_csv)) as csvfile:
            # data = list(csv.reader(csvfile))
            reader = csv.reader(x.replace('\0', '') for x in csvfile)
            data = list(reader)
        with open(str(latest_csv), 'w', newline='') as csvfile:
            wr = csv.writer(csvfile)
            wr.writerow(data[0])
            for i in list(range(1, -1, -1)):
                if i != 0:
                    wr.writerow(data[-i])
    except:
        error_ind = True
        print('Error detected: Openface')
    ####################################################################################################################

    # MAIN LOOP WITH SAFETY CHECK
    while fly and all(qcf.is_safe() for qcf in qcfs):

        # Land with Esc
        if last_key_pressed == pynput.keyboard.Key.esc:
            print(f'[t={int(dt)}] Escape by keyboard.')
            break

        # GUI input
        mode = game_mgr.input()

        # Mind the clock
        dt = time() - t

        # Mission completed
        if all(mission_complete):
            print(f'[t={int(dt)}] All missions completed!')
            break

        # Time out for safety
        for idx, qcf in enumerate(qcfs):
            # GUI: transform from meter to gui
            pos = [qcfs[idx].pose.x, qcfs[idx].pose.y]
            game_mgr.objects[idx].position = game_mgr.position_meter_to_gui_single(pos)
            yaw = -math.atan2(qcfs[idx].pose.rotmatrix[1][0], qcfs[idx].pose.rotmatrix[0][0])
            game_mgr.objects[idx].rt = yaw * 180 / np.pi
            game_mgr.objects[idx + 2].position[1] = game_mgr.altitude_meter_to_gui(qcfs[idx].pose.z, noise=False)

            # Initial hover
            if dt < hover_duration:
                target = Pose(takeoff_positions[idx][0], takeoff_positions[idx][1], 0.5 * world.expanse[2])
                qcf.safe_position_setpoint(target)
                sleep(0.01)

            # Proceed target acquisition
            elif dt < total_duration:
                # Move to the current target
                target_current = drone_paths[idx][target_index[idx] + 1]
                # Mission completed?
                if mission_complete[idx]:
                    qcf.land_in_place()
                    sleep(0.01)
                    continue
                else:
                    # Current position check for mutual avoidance
                    drone_positions = [[qcfs[0].pose.x, qcfs[0].pose.y], [qcfs[1].pose.x, qcfs[1].pose.y]]
                    # Go to target with adjusted altitude
                    altitude_adjust = avoid_altitude_control(drone_positions)
                    # Temporary
                    current_trajectory = drone_trajectory[idx][target_index[idx]]
                    # Find the target positions
                    increment = speed_constant * (dt - dt_prev[idx])
                    path_index[idx] += increment
                    picked = min(int(path_index[idx]) + 1, len(current_trajectory))
                    target = Pose(current_trajectory[-picked][0], current_trajectory[-picked][1],
                                  0.5 * world.expanse[2] + altitude_adjust[idx])
                    qcf.safe_position_setpoint(target)
                    # Time
                    dt_prev[idx] = dt
                    sleep(0.01)

                # Check distance
                target_distance[idx] = distance(target_current, [qcf.pose.x, qcf.pose.y])

                # Check stay start time
                if target_distance[idx] < stay_distance:
                    # Measure stay time
                    if stay_flag[idx]:
                        # Measure: start point
                        start_time[idx] = time()
                        stay_flag[idx] = False
                    # Measure: end point
                    stay_time[idx] = time() - start_time[idx]

                # Check stay duration
                if stay_time[idx] > stay_duration:
                    if target_index[idx] < len(drone_trajectory[idx]) - 1:
                        # Generate victim
                        if not game_mgr.victim_detected[idx]:
                            game_mgr.victim_id[idx] = survivor_images[survivor_index]
                            survivor_index += 1
                            game_mgr.victim_detected[idx] = True
                            # To record response time
                            game_mgr.victim_timing[idx] = dt

                        # Once victim is selected, close it: only if there is no unassigned target
                        if game_mgr.target_decided:
                            if game_mgr.victim_clicked[idx]:
                                game_mgr.victim_id[idx] = 0
                                game_mgr.victim_detected[idx] = False
                                # To record response time
                                game_mgr.missions[idx].response_time.append(dt - game_mgr.victim_timing[idx])
                                game_mgr.missions[idx].time_stamp.append(dt)
                                game_mgr.missions[idx].drone_id.append(idx)
                                game_mgr.victim_timing[idx] = 0
                                print(f'Response time by drone {int(idx + 1)}: {game_mgr.missions[idx].response_time[-1]:.2f} sec')
                                # Print status
                                print(f'[t={int(dt)}] Drone {int(idx + 1)}: target {int(target_index[idx] + 1)} accomplished')
                                target_remaining.remove(target_current)
                                # Update and reset
                                target_index[idx] += 1
                                stay_flag[idx] = True
                                stay_time[idx] = 0
                                # Temporary
                                path_index[idx] = 0
                        else:
                            game_mgr.victim_block_choice[idx] = True
                    else:
                        mission_complete[idx] = True

                # Mission (target) update: triggered by the number of remaining targets
                if idx == 0:  # Do only once: 1st drone
                    if len(target_remaining) == 4 and dt > update_on and not game_mgr.new_target_triggered:
                        # New target detected (comm from mission control)
                        print(f'[t={int(dt)}] New target identified')
                        new_target_gui = game_mgr.position_meter_to_gui_single(new_target)
                        game_mgr.set_target(new_target=[new_target_gui])
                        # Update remaining
                        target_remaining.append(new_target)
                        # Close the case
                        game_mgr.new_target_triggered = True
                        game_mgr.target_decided = False

                        # Compute distance with additional target
                        new_start_positions = [drone_paths[0][target_index[0] + 1],
                                               drone_paths[1][target_index[1] + 1]]
                        # [Temporary: code B]
                        target_remaining_copy = target_remaining.copy()
                        target_remaining_copy.remove(new_start_positions[0])
                        target_remaining_copy.remove(new_start_positions[1])
                        # Two re-planning scenarios
                        output1, path1 = assign_targets_with_path_lengths(new_start_positions,
                                                                          target_remaining_copy, 0,
                                                                          landing=takeoff_positions)
                        output2, path2 = assign_targets_with_path_lengths(new_start_positions,
                                                                          target_remaining_copy, 1,
                                                                          landing=takeoff_positions)
                        game_mgr.planning_distances = output1 + output2

                    # Decision-making on the new target (triggered) by mouse action
                    # [Temporary] Function allocation
                    if fa == 1:
                        game_mgr.target_clicked = 2
                    if game_mgr.new_target_triggered and not game_mgr.target_decided and game_mgr.target_clicked:
                        print(f'[t={int(dt)}] Decision made on the new target')
                        # Restore block if there is any
                        game_mgr.victim_block_choice = [False, False]
                        # Target decision made
                        game_mgr.target_decided = True
                        # Reset the planning distance to zeros
                        game_mgr.planning_distances = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        # Selecting one re-planning scenario
                        if game_mgr.target_clicked == 1:
                            drone_paths = path1
                        elif game_mgr.target_clicked == 2:
                            drone_paths = path2
                        new_drone_trajectory = [[], []]
                        for d_inx in range(2):
                            for p_idx in range(len(drone_paths[d_inx]) - 1):
                                rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx],
                                                                  drone_paths[d_inx][p_idx + 1],
                                                                  0.08, 0.05, 5000)
                                rrt_conn.planning()
                                rrt_conn.smoothing()
                                new_drone_trajectory[d_inx].append(rrt_conn.path)
                        # Maintain the current trajectory
                        drone_trajectory = [[drone_trajectory[0][target_index[0]]],
                                            [drone_trajectory[1][target_index[1]]]]
                        for i in range(2):
                            for j in range(len(new_drone_trajectory[i])):
                                drone_trajectory[i].append(new_drone_trajectory[i][j])
                        target_index = [0, 0]
                        # [Temporary] Recover the overlapped start positions (target_current)
                        drone_paths[0].insert(0, new_start_positions[0])
                        drone_paths[1].insert(0, new_start_positions[1])

                # Wind (fault) condition
                if idx == 0:
                    # Trigger windy condition based on time
                    if not game_mgr.wind_triggered and dt > wind_on:
                        game_mgr.wind_danger = True
                        game_mgr.wind_triggered = True
                        game_mgr.wind_closed = False
                        game_mgr.wind_decided = False
                        print(f'[t={int(dt)}] Environmental change: dangerous wind')
                        speed_constant = 1.0
                        # Put obstacle avoidance here
                        game_mgr.set_wind(wind, meter=True)

                    # Turn-off windy condition based on time
                    if game_mgr.wind_danger and game_mgr.wind_triggered and dt > wind_off:
                        game_mgr.wind_danger = False
                        game_mgr.wind_closed = True
                        game_mgr.wind_decided = True
                        print(f'[t={int(dt)}] Environmental change: stable wind')
                        # Remove wind graphic
                        game_mgr.reset_wind()
                        # Speed adjustment
                        speed_constant = 3.0
                        # Task allocation
                        current_start = [[qcfs[0].pose.x, qcfs[0].pose.y], [qcfs[1].pose.x, qcfs[1].pose.y]]
                        drone_paths = assign_targets_to_drones(current_start, target_remaining,
                                                               landing=takeoff_positions)
                        # RRT-connect for all drone paths
                        drone_trajectory = [[], []]
                        for d_inx in range(2):
                            for p_idx in range(len(drone_paths[d_inx]) - 1):
                                rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx],
                                                                  drone_paths[d_inx][p_idx + 1], 0.08, 0.05,
                                                                  5000)
                                rrt_conn.utils.update_obs([], [], [])
                                rrt_conn.planning()
                                rrt_conn.smoothing()
                                drone_trajectory[d_inx].append(rrt_conn.path)
                        target_index = [0, 0]
                        # Time correction
                        path_index = [0, 0]
                        dt_prev[0] = dt
                        dt_prev[1] = dt

                    # [Temporary] Function allocation
                    # If fa = 1 or 2, skip the wind change response
                    if fa == 1 or fa == 2:
                        game_mgr.wind_clicked = 1
                    # Ask for decision
                    # If fa = 3, ask for human's decision
                    if game_mgr.wind_danger and game_mgr.wind_triggered and not game_mgr.wind_decided:
                        if game_mgr.wind_clicked == 1:
                            game_mgr.wind_closed = True
                            # Distance condition
                            d1 = distance([qcfs[0].pose.x, qcfs[0].pose.y], wind[0:2])
                            d2 = distance([qcfs[1].pose.x, qcfs[1].pose.y], wind[0:2])
                            # When wind is not overlapped, re-plan trajectory
                            if d1 > wind[2] * 1.1 and d2 > wind[2] * 1.1:
                                print(f'[t={int(dt)}] Change routes')
                                game_mgr.wind_decided = True
                                # Speed adjustment
                                speed_constant = 3.0
                                # Task allocation
                                current_start = [[qcfs[0].pose.x, qcfs[0].pose.y], [qcfs[1].pose.x, qcfs[1].pose.y]]
                                drone_paths = assign_targets_to_drones(current_start, target_remaining,
                                                                       landing=takeoff_positions)
                                # RRT-connect for all drone paths
                                drone_trajectory = [[], []]
                                for d_inx in range(2):
                                    for p_idx in range(len(drone_paths[d_inx]) - 1):
                                        rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx],
                                                                          drone_paths[d_inx][p_idx + 1], 0.08,
                                                                          0.05, 5000)
                                        rrt_conn.utils.update_obs([wind], [], [])
                                        rrt_conn.planning()
                                        rrt_conn.smoothing()
                                        drone_trajectory[d_inx].append(rrt_conn.path)
                                target_index = [0, 0]
                                # Time correction
                                path_index = [0, 0]
                                dt_prev[0] = dt
                                dt_prev[1] = dt

                        elif game_mgr.wind_clicked == 2:
                            game_mgr.wind_closed = True
                            game_mgr.wind_decided = True
                            print(f'[t={int(dt)}] Maintain routes')

                # Mission report
                if idx == 0:
                    if not game_mgr.report_triggered and dt > report_on:
                        game_mgr.report_triggered = True
                        game_mgr.report_requested = True
                    if game_mgr.report_requested and game_mgr.report_clicked:
                        game_mgr.report_requested = False

            else:
                fly = False

        # GUI rendering
        game_mgr.update()
        game_mgr.render()

    # Land till the end
    while max(qcfs[0].pose.z, qcfs[1].pose.z) > 0.1:
        for idx, qcf in enumerate(qcfs):
            qcf.land_in_place()
            sleep(0.01)
            pos = [qcfs[idx].pose.x, qcfs[idx].pose.y]
            game_mgr.objects[idx].position = game_mgr.position_meter_to_gui_single(pos)
            yaw = -math.atan2(qcfs[idx].pose.rotmatrix[1][0], qcfs[idx].pose.rotmatrix[0][0])
            game_mgr.objects[idx].rt = yaw * 180 / np.pi
            game_mgr.objects[idx + 2].position[1] = game_mgr.altitude_meter_to_gui(qcfs[idx].pose.z, noise=False)
        # GUI rendering
        game_mgr.update()
        game_mgr.render()

    ####################################################################################################################
    # Game end - data processing
    try:
        shutil.copyfile(latest_csv, dataset_directory + 'test_result_of_' + time_string + '_' + participant + '.csv')
        shutil.copyfile(ecg_csv, dataset_directory + 'test_result_ecg_' + time_string + '_' + participant + '.csv')
        print('data copied')
    except shutil.SameFileError:
        print('Same File Error')
    except NameError:
        print('Name Error')
    ####################################################################################################################

    # Self-report
    game_mgr.mode = 3
    while game_mgr.mode == 3:
        game_mgr.workload_render()
    while game_mgr.mode == 4:
        game_mgr.perceived_risk_render()

    # Data out: {response time, correctness, workload survey}
    time_stamp = game_mgr.missions[0].time_stamp + game_mgr.missions[1].time_stamp
    drone_id = game_mgr.missions[0].drone_id + game_mgr.missions[1].drone_id
    response = game_mgr.missions[0].response_time + game_mgr.missions[1].response_time
    correctness = game_mgr.missions[0].correctness + game_mgr.missions[1].correctness
    workload = game_mgr.workload
    p_risk = game_mgr.p_risk

    # If empty, skip
    if not response:
        time_stamp = [0]
        drone_id = [0]
        response = [0]
        correctness = [0]

    filename = dataset_directory + 'human_log_' + time_string + '_' + participant + '.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Drone', 'Time', 'Response', 'Correctness', 'Workload', 'Risk', 'Allocation'])
        for i in range(len(response)):
            writer.writerow([drone_id[i], time_stamp[i], response[i], correctness[i], workload, p_risk, fa])

    # Log stop
    stop_event.set()
    print('Stop logging.')
    print('End of Experiment.')
