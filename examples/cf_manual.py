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
import pygame

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

# Joystick input
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)


def joystick_to_command(joystick_, rr=30, rp=30, ry=200, rt=None):
    """
    - Axes mapping depends on your joystick!
    - .get_axis method required (from pygame)
    Joystick range is given
    rr: range_roll = 30     # -X to X, deg
    rp: range_pitch = 30    # -X to X, deg
    ry: range_yaw (rate) = 200     # -X to X, deg/s
    rt: range_thrust = [10000, 60000]
    """
    if rt is None:
        rt = [10000, 60000]

    # Axis mapping
    roll = joystick_.get_axis(2)
    pitch = joystick_.get_axis(3)
    yaw = joystick_.get_axis(0)
    thrust = joystick_.get_axis(1)

    # Axis to command transform
    roll = rr * roll
    pitch = -rp * pitch
    yaw = ry * yaw
    thrust = int(max((rt[0] - rt[1]) * thrust + rt[0], rt[0]))
    return roll, pitch, yaw, thrust


# Prepare for liftoff
with QualisysCrazyflie(cf_body_name,
                       cf_uri,
                       world,
                       marker_ids=cf_marker_ids,
                       qtm_ip="192.168.123.2") as qcf:

    # Let there be time
    t = time()
    dt = 0

    print("Beginning maneuvers...")

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

        # Joystick input
        pygame.event.get()
        [cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust] = joystick_to_command(joystick)

        # Mind the clock
        dt = time() - t

        # Calculate Crazyflie's angular position in circle, based on time
        phi = circle_speed_factor * dt * 360

        # Safety check
        if not qcf.is_safe():
            print(f'Unsafe! {str(qcf.pose)}')

        # Unlock startup thrust protection (what is the minimum required time? current 3 secs)
        if dt < 3:
            qcf.cf.commander.send_setpoint(0, 0, 0, 0)

        elif dt < 60:
            qcf.cf.commander.send_setpoint(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)

        else:
            break

    # Land
    while qcf.pose.z > 0.1:
        qcf.land_in_place()

    # Data logging close
    lg_stab.stop()
    print('Logging Finished.')

    # Joystick/PyGame quit
    pygame.quit()
    print('Quit PyGame.')
