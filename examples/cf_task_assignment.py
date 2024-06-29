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

    # MAIN LOOP WITH SAFETY CHECK
    while fly and all(qcf.is_safe() for qcf in qcfs):

        # Land with Esc
        if last_key_pressed == pynput.keyboard.Key.esc:
            break

        # Mind the clock
        dt = time() - t

        # Cycle all drones
        for idx, qcf in enumerate(qcfs):

            # Take off and hover in the center of safe airspace
            if dt < 3:
                # print(f'[t={int(dt)}] Maneuvering - Center...')
                # Set target
                x = np.interp(idx,
                              [0,
                               len(qcfs) - 1],
                              [world.origin.x - world.expanse[0] / 2,
                                  world.origin.x + world.expanse[0] / 2])
                target = Pose(x,
                              world.origin.y,
                              world.expanse[2] * 0.5)
                # Engage
                qcf.safe_position_setpoint(target)
                sleep(0.01)

            # Move out half of the safe airspace in the X direction and circle around Z axis
            elif dt < 30:
                # print(f'[t={int(dt)}] Maneuvering - Circle around Z...')
                # Set target
                phi = (dt * 90) % 360  # Calculate angle based on time
                # Offset angle based on array
                phi = phi + 360 * (idx / len(qcfs))
                _x, _y = utils.pol2cart(0.6, phi)
                target = Pose(world.origin.x + _x,
                              world.origin.y + _y,
                              world.expanse[2] * 0.5 * (idx + 1.0) * 0.5)
                # Engage
                qcf.safe_position_setpoint(target)
                sleep(0.01)

            # Back to center
            elif dt < 33:
                # print(f'[t={int(dt)}] Maneuvering - Center...')
                # Set target
                x = np.interp(idx,
                              [0,
                               len(qcfs) - 1],
                              [world.origin.x - world.expanse[0] / 2,
                                  world.origin.x + world.expanse[0] / 2])
                target = Pose(x,
                              world.origin.y,
                              world.expanse[2] * 0.5 * (idx + 1.0) * 0.5)
                # Engage
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
