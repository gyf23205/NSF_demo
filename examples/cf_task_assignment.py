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

from rrt_2D import rrt_star, rrt_connect

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
hover_duration = 5
stay_duration = 5
stay_distance = 0.1
separation_distance = 0.5
total_duration = 60
speed_constant = 3.0

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
    # Let there be time
    t = time()
    dt = 0

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

    # Initialize
    target_index = [0, 0]
    target_distance = [0, 0]

    # Stay flag
    stay_flag = [True, True]
    stay_time = [0, 0]
    start_time = [0, 0]

    # Mission completion
    mission_complete = [False, False]

    # MAIN LOOP WITH SAFETY CHECK
    while fly and all(qcf.is_safe() for qcf in qcfs):

        # Land with Esc
        if last_key_pressed == pynput.keyboard.Key.esc:
            break

        # Mind the clock
        dt = time() - t

        # Mission completed
        if all(mission_complete):
            print(f'[t={int(dt)}] All missions completed!')
            break

        # Time out for safety
        for idx, qcf in enumerate(qcfs):
            # Initial hover
            if dt < hover_duration:
                target = Pose(takeoff_positions[idx][0], takeoff_positions[idx][1], 0.5 * world.expanse[2])
                # Engage
                qcf.safe_position_setpoint(target)
                sleep(0.01)

            # Proceed target acquisition
            elif dt < total_duration:
                # Move to the current target
                target_current = drone_paths[idx][target_index[idx]]
                # Mission completed?
                if mission_complete[idx]:
                    qcf.land_in_place()
                    sleep(0.01)
                    continue
                else:
                    # Current position check for mutual avoidance
                    drone_positions = [[qcfs[0].pose.x, qcfs[0].pose.y], [qcfs[1].pose.x, qcfs[1].pose.y]]
                    altitude_adjust = avoid_altitude_control(drone_positions)
                    # Go to target with adjusted altitude
                    target = Pose(target_current[0], target_current[1], 0.5 * world.expanse[2] + altitude_adjust[idx])
                    qcf.safe_position_setpoint(target)
                    sleep(0.01)

                # Check distance
                target_distance[idx] = distance(target_current, [qcf.pose.x, qcf.pose.y])

                # Check stay start time
                if target_distance[idx] < stay_distance:
                    if stay_flag[idx]:
                        start_time[idx] = time()
                        stay_flag[idx] = False
                    stay_time[idx] = time() - start_time[idx]

                # Check stay duration
                if stay_time[idx] > stay_duration:
                    if target_index[idx] < len(drone_paths[idx]) - 1:
                        target_index[idx] += 1
                        stay_flag[idx] = True
                        stay_time[idx] = 0
                    else:
                        mission_complete[idx] = True

            else:
                fly = False

    # Land till the end
    while max(qcfs[0].pose.z, qcfs[1].pose.z) > 0.1:
        for idx, qcf in enumerate(qcfs):
            qcf.land_in_place()
            sleep(0.01)

    # Log stop
    stop_event.set()
    print('Stop logging.')
