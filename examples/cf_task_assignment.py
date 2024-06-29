"""
qfly | Qualisys Drone SDK based Code
https://github.com/qualisys/qualisys_drone_sdk

Task Allocations with Multiple Drones
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
hover_time = 10
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


def assign_targets_to_drones(drones, targets):
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

    return paths


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
    # Task allocation example
    target_positions = [[2.0, -0.7], [-0.1, -0.9], [-2.0, -1.0], [-1.0, 0.7], [2.0, 1.0]]
    drone_positions = [[qcfs[0].pose.x, qcfs[0].pose.y], [qcfs[1].pose.x, qcfs[1].pose.y]]
    drone_paths = assign_targets_to_drones(drone_positions, target_positions)
    #################################################################

    # MAIN LOOP WITH SAFETY CHECK
    while fly and all(qcf.is_safe() for qcf in qcfs):

        # Land with Esc
        if last_key_pressed == pynput.keyboard.Key.esc:
            break

        # Mind the clock
        dt = time() - t

        # Cycle all drones
        for idx, qcf in enumerate(qcfs):

            if dt < 40:
                # Determine the current index based on elapsed time
                t_idx = min(int(dt // 10), len(drone_paths[idx]) - 1)
                # Process the current position
                position = drone_paths[idx][t_idx]
                target = Pose(position[0], position[1], 0.5 * world.expanse[2] * (idx + 1.0) * 0.5)
                qcf.safe_position_setpoint(target)
                sleep(0.01)

            else:
                fly = False

    # Land (why qcf.pose.z not qcfs?)
    while qcf.pose.z > 0.1:
        for idx, qcf in enumerate(qcfs):
            qcf.land_in_place()
            sleep(0.01)

    # Log stop
    stop_event.set()
    print('Stop logging.')
