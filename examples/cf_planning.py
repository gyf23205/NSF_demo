"""
qfly | Qualisys Drone SDK based Code
https://github.com/qualisys/qualisys_drone_sdk

Path Planning with RRT* Variations - Crazyflie
Edited by Sooyung Byeon, Purdue University
June, 2024

ESC to land at any time.
"""
import pynput
from time import sleep, time, strftime
from qfly import Pose, QualisysCrazyflie, World, utils

import logging
from cflib.crazyflie.log import LogConfig
import csv

from rrt_2D import rrt_star, rrt_connect

# Drone Setting: name and address
cf_body_name = 'nsf11'                  # QTM rigid body name
cf_uri = 'radio://0/80/2M/E7E7E7E711'   # Crazyflie address
cf_marker_ids = [11, 12, 13, 14]        # Active marker IDs

# Drone Setting: Physical constraints
hover_time = 10
speed_constant = 3.0

# World Setting: the World object comes with sane defaults
world = World()

# Log Setting: Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# Log Setting: File name and directory
time_string = strftime('%y%m%d%H%M%S')
log_name = '../logs/log' + time_string + '.csv'

# Log Setting: log configuration
#########################################################
lg_stab = LogConfig(name='Standard', period_in_ms=100)
lg_stab.add_variable('stateEstimate.x', 'FP16')
lg_stab.add_variable('stateEstimate.y', 'FP16')
lg_stab.add_variable('stateEstimate.z', 'FP16')
lg_stab.add_variable('stateEstimate.vx', 'FP16')
lg_stab.add_variable('stateEstimate.vy', 'FP16')
lg_stab.add_variable('stateEstimate.vz', 'FP16')
lg_stab.add_variable('stateEstimate.roll', 'FP16')
lg_stab.add_variable('stateEstimate.pitch', 'FP16')
lg_stab.add_variable('stateEstimate.yaw', 'FP16')
lg_stab.add_variable('controller.cmd_roll', 'FP16')
lg_stab.add_variable('controller.cmd_pitch', 'FP16')
lg_stab.add_variable('controller.cmd_yaw', 'FP16')
lg_stab.add_variable('controller.cmd_thrust', 'FP16')
#########################################################
# Open logging csv file: put head (variable names)
with open(log_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['timestamp'] + [variable.name for variable in lg_stab.variables])


# Log line by line in the opened csv file
def log_stab_callback(timestamp, data, logconf):
    # print('[%d][%s]: %s' % (timestamp, logconf.name, data))
    with open(log_name, 'a', newline='') as file_:
        writer_line = csv.writer(file_)
        writer_line.writerow([timestamp] + list(data.values()))


# Set up keyboard callback
def on_press(key):
    """React to keyboard."""
    global last_key_pressed
    last_key_pressed = key
    if key == pynput.keyboard.Key.esc:
        fly = False


# Keyboard input: Watch key presses with a global variable
last_key_pressed = None

# Listen to the keyboard
listener = pynput.keyboard.Listener(on_press=on_press)
listener.start()


# Prepare for liftoff
with QualisysCrazyflie(cf_body_name,
                       cf_uri,
                       world,
                       marker_ids=cf_marker_ids,
                       qtm_ip="192.168.123.2") as qcf:
    sleep(1.0)
    print("Beginning maneuvers...")

    ######################
    # Path planning: RRT*
    x_start = (qcf.pose.x, qcf.pose.y)  # Starting node
    x_goal = (1.7, 0.9)  # Goal node
    # rrt_star = rrt_star.RrtStar(x_start, x_goal, 0.1, 0.10, 0.2, 5000)
    rrt_star = rrt_connect.RrtConnect(x_start, x_goal, 0.08, 0.05, 5000)
    rrt_star.planning()
    rrt_star.smoothing()
    # print(rrt_star.path)
    path_point = len(rrt_star.path)
    path_index = 0
    ######################

    # Let there be time
    t = time()
    dt = 0

    ######################
    # Logging start
    qcf.cf.log.add_config(lg_stab)
    lg_stab.data_received_cb.add_callback(log_stab_callback)
    lg_stab.start()
    print("Logging Start...")
    sleep(1.0)
    ######################

    # Take-off position
    take_off_position = [qcf.pose.x, qcf.pose.y]

    # MAIN LOOP WITH SAFETY CHECK
    while qcf.is_safe():
        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            break

        # Mind the clock
        dt = time() - t

        # Unlock startup thrust protection (what is the minimum required time? current 3 secs)
        if dt < hover_time:
            # Set target
            target = Pose(take_off_position[0], take_off_position[1], 1.0)
            # Engage
            qcf.safe_position_setpoint(target)
            sleep(0.02)

        elif dt < 90:
            path_index = int(speed_constant * (dt - hover_time) + 1)
            if path_index < path_point + 1:
                target = Pose(rrt_star.path[-path_index][0], rrt_star.path[-path_index][1], 1.0)
                qcf.safe_position_setpoint(target)
                sleep(0.02)
            else:
                print(f'[t={int(dt)}] Target Reached.')
                break

        else:
            print(f'[t={int(dt)}] Time-out.')
            break

    # Land
    while qcf.pose.z > 0.1:
        qcf.land_in_place()

    # Data logging close
    lg_stab.stop()
    print('Logging Finished.')
