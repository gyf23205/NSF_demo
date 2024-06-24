"""
qfly | Qualisys Drone SDK based Code
https://github.com/qualisys/qualisys_drone_sdk

Manual Control and Data Logging - Crazyflie
Edited by Sooyung Byeon, Purdue University
June, 2024

Xbox controller required
ESC to land at any time.
"""
import pynput
from time import sleep, time, strftime
from qfly import Pose, QualisysCrazyflie, World, utils

import logging
from cflib.crazyflie.log import LogConfig
import csv

from rrt_2D import rrt_star

# Drone Setting: name and address
cf_body_name = 'nsf11'                  # QTM rigid body name
cf_uri = 'radio://0/80/2M/E7E7E7E711'   # Crazyflie address
cf_marker_ids = [11, 12, 13, 14]        # Active marker IDs

# Drone Setting: Physical constraints
circle_radius = 0.5                     # Radius of the circular flight path
circle_speed_factor = 0.12              # How fast the Crazyflie should move along circle

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
    x_goal = (1.8, 0.9)  # Goal node
    rrt_star = rrt_star.RrtStar(x_start, x_goal, 1, 0.10, 2, 6000)
    rrt_star.planning()
    n_point = len(rrt_star.path)
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

    # MAIN LOOP WITH SAFETY CHECK
    while qcf.is_safe():
        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            break

        # Mind the clock
        dt = time() - t

        # Safety check
        if not qcf.is_safe():
            print(f'Unsafe! {str(qcf.pose)}')

        # Unlock startup thrust protection (what is the minimum required time? current 3 secs)
        if dt < 5:
            # Set target
            target = Pose(qcf.pose.x, qcf.pose.y, 1.0)
            # Engage
            qcf.safe_position_setpoint(target)
            # Temporary time stamp
            n_count = 0
            n_current = 0

        else:
            break

    # Land
    while qcf.pose.z > 0.1:
        qcf.land_in_place()

    # Data logging close
    lg_stab.stop()
    print('Logging Finished.')
