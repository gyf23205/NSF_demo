"""
qfly | Qualisys Drone SDK based Code
https://github.com/qualisys/qualisys_drone_sdk

Task Allocations with Multiple Drones: nearest neighbor algorithm
Including Motion Planning by RRT* Variations - Crazyflie
Edited by Sooyung Byeon, Purdue University
June, 2024

ESC to land at any time.
"""

import pynput
from time import sleep, time, strftime
from qfly import Pose, QualisysCrazyflie, World, ParallelContexts, utils
import csv
import numpy as np
import math
import threading
import matplotlib.pyplot as plt

from rrt_2D import rrt_connect

import game

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

# Drone Setting: Physical constraints
hover_duration = 10
stay_duration = 5
stay_distance = 0.1
separation_distance = 0.5
total_duration = 120
speed_constant = 5.0
indexing_time = [hover_duration, hover_duration]

# Set up world - the World object comes with sane defaults (geofencing)
world = World()

# Watch key presses with a global variable
last_key_pressed = None

# Log Setting: File name and directory
time_string = strftime('%y%m%d%H%M%S')
log_name = '../logs/log' + time_string + '.csv'

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


def assign_targets_to_drones(drones, targets, go_back=True):
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

    if go_back:
        for i in range(len(drones)):
            paths[i].append(drones[i])

    return paths


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

    #################################################################
    # Task allocation: mission and computation
    target_positions = [[2.0, -0.7], [-0.1, -0.9], [-2.0, -1.0], [-1.0, 0.7], [2.0, 1.0], [-0.5, -0.5]]
    takeoff_positions = [[qcfs[0].pose.x, qcfs[0].pose.y], [qcfs[1].pose.x, qcfs[1].pose.y]]
    drone_paths = assign_targets_to_drones(takeoff_positions, target_positions)
    #################################################################

    #################################################################
    # RRT connect for all drone paths (targets)
    drone_trajectory = [[], []]
    for d_inx in range(2):
        for p_idx in range(len(drone_paths[d_inx]) - 1):
            rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx], drone_paths[d_inx][p_idx + 1], 0.08, 0.05, 5000)
            rrt_conn.planning()
            rrt_conn.smoothing()
            drone_trajectory[d_inx].append(rrt_conn.path)

    # Temporary: verification plot
    # colors = ['blue', 'green', 'red', 'yellow', 'cyan', 'black', 'magenta']
    # markers = ['o', 'x']
    # for i in range(2):
    #     for j, sublist in enumerate(drone_trajectory[i]):
    #         x_values = [pair[0] for pair in sublist]
    #         y_values = [pair[1] for pair in sublist]
    #         plt.scatter(x_values, y_values, color=colors[j], marker=markers[i], label=f't[{i}]')
    #         print(f'[Drone {int(i + 1)}]: Trajectory {int(j + 1)}: Length {len(sublist)}.')
    # plt.show()
    #################################################################

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

    # Testing
    target_gui = np.array(target_positions)
    for k in range(len(target_positions)):
        target_gui[k][0] = 300.0 * target_gui[k][0] + 750
        target_gui[k][1] = -300.0 * target_gui[k][1] + 450
    takeoff_gui = np.array(takeoff_positions)
    for k in range(len(takeoff_positions)):
        takeoff_gui[k][0] = 300.0 * takeoff_gui[k][0] + 750
        takeoff_gui[k][1] = -300.0 * takeoff_gui[k][1] + 450
    game_mgr.set_target(target=target_gui)
    game_mgr.set_takeoff_positions(takeoff_gui)

    # Let there be time (This should be right before the while loop)
    t = time()
    dt = 0

    # MAIN LOOP WITH SAFETY CHECK
    while fly and all(qcf.is_safe() for qcf in qcfs):

        # Land with Esc
        if last_key_pressed == pynput.keyboard.Key.esc:
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
            # GUI
            game_mgr.objects[idx].position[0] = qcfs[idx].pose.x * 300.0 + 750
            game_mgr.objects[idx].position[1] = -qcfs[idx].pose.y * 300.0 + 450
            yaw = -math.atan2(qcfs[idx].pose.rotmatrix[1][0], qcfs[idx].pose.rotmatrix[0][0])
            game_mgr.objects[idx].rt = yaw * 180 / np.pi
            game_mgr.objects[idx + 2].position[1] = -75.0 * qcfs[idx].pose.z + 200.0

            # Initial hover
            if dt < hover_duration:
                target = Pose(takeoff_positions[idx][0], takeoff_positions[idx][1], 0.5 * world.expanse[2])
                # Engage
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
                    path_index = min(int(speed_constant * (dt - indexing_time[idx])) + 1, len(current_trajectory))
                    target = Pose(current_trajectory[-path_index][0], current_trajectory[-path_index][1],
                                  0.5 * world.expanse[2] + altitude_adjust[idx])
                    qcf.safe_position_setpoint(target)
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
                            game_mgr.victim_id[idx] = np.random.randint(low=1, high=21)
                            game_mgr.victim_detected[idx] = True

                        # Once victim is selected, close it: only if there is no unassigned target
                        if game_mgr.target_decided:
                            if game_mgr.victim_clicked[idx]:
                                game_mgr.victim_id[idx] = 0
                                game_mgr.victim_detected[idx] = False
                                # Print status
                                print(
                                    f'[t={int(dt)}] Drone {int(idx + 1)}: target {int(target_index[idx] + 1)} accomplished')
                                # target_remaining.remove(target_current)
                                # Update and reset
                                target_index[idx] += 1
                                stay_flag[idx] = True
                                stay_time[idx] = 0
                                # Temporary
                                indexing_time[idx] = dt
                        else:
                            game_mgr.victim_block_choice[idx] = True
                    else:
                        mission_complete[idx] = True

                # if stay_time[idx] > stay_duration:
                #     # Print status
                #     print(f'[t={int(dt)}] Drone {int(idx + 1)}: target {int(target_index[idx] + 1)} stay checked')
                #     if target_index[idx] < len(drone_trajectory[idx]) - 1:
                #         # Update and reset
                #         target_index[idx] += 1
                #         stay_flag[idx] = True
                #         stay_time[idx] = 0
                #         # Temporary
                #         indexing_time[idx] = dt
                #     else:
                #         mission_complete[idx] = True

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
            # This should be a function
            game_mgr.objects[idx].position[0] = qcfs[idx].pose.x * 300.0 + 750
            game_mgr.objects[idx].position[1] = -qcfs[idx].pose.y * 300.0 + 450
            yaw = -math.atan2(qcfs[idx].pose.rotmatrix[1][0], qcfs[idx].pose.rotmatrix[0][0])
            game_mgr.objects[idx].rt = yaw * 180 / np.pi
            game_mgr.objects[idx + 2].position[1] = -75.0 * qcfs[idx].pose.z + 200.0
            # GUI rendering
        game_mgr.update()
        game_mgr.render()

    # Log stop
    stop_event.set()
    print('Stop logging.')
