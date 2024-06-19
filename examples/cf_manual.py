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

# Joystick range
range_roll = 30     # -X to X, deg
range_pitch = 30    # -X to X, deg
range_yaw = 200     # -X to X, deg/s
range_thrust = [10000, 60000]

# Joystick range transform function (TBD)


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
    qcf.cf.log.add_config(lg_stab)
    lg_stab.data_received_cb.add_callback(log_stab_callback)
    lg_stab.start()
    print("Logging Start...")
    sleep(1.0)
    ######################

    # Unlock startup thrust protection
    qcf.cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)
    sleep(1)

    # MAIN LOOP WITH SAFETY CHECK
    while qcf.is_safe():
        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            break

        # Joystick input
        pygame.event.get()
        cmd_yaw = joystick.get_axis(0)
        cmd_thrust = joystick.get_axis(1)
        cmd_roll = joystick.get_axis(2)
        cmd_pitch = joystick.get_axis(3)

        # Temporary transform
        cmd_roll = range_roll * cmd_roll
        cmd_pitch = -range_pitch * cmd_pitch
        cmd_yaw = range_yaw * cmd_yaw
        cmd_thrust = int(max((range_thrust[0] - range_thrust[1]) * cmd_thrust + range_thrust[0], range_thrust[0]))
        # print([cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust])

        # Mind the clock
        dt = time() - t

        # Calculate Crazyflie's angular position in circle, based on time
        phi = circle_speed_factor * dt * 360

        # Take off and hover in the center of safe airspace for 5 seconds
        # if dt < 5:
        #     # print(f'[t={int(dt)}] Maneuvering - Center...')
        #     # Set target
        #     target = Pose(world.origin.x, world.origin.y, world.expanse)
        #     # Engage
        #     qcf.safe_position_setpoint(target)

        # Thrust input initialization (documented in user-guides/python_api)
        # qcf.cf.commander.send_setpoint(0.0, 0.0, 0.0, 0)

        # Take off and hover in the center of safe airspace for 5 seconds
        # if dt < 10:
        #     target = Pose(0.0, 0.0, 0.5)
        #     qcf.safe_position_setpoint(target)

        # Manual control temporary test
        # if dt < 10:
        #     target = Pose(0.0, 0.0, 0.0)
        #     qcf.safe_position_setpoint(target)

        if dt < 3:
            # target = Pose(0.0, 0.0, 0.2)
            # qcf.safe_position_setpoint(target)
            qcf.cf.commander.send_setpoint(0, 0, 0, 0)

        elif dt < 30:
            qcf.cf.commander.send_setpoint(cmd_roll, cmd_pitch, cmd_yaw, cmd_thrust)

        # elif dt < 20:
        #     if qcf.pose.z < 0.9:
        #         print(f'[z={int(dt)}] Thrust!')
        #         qcf.cf.commander.send_setpoint(0.0, 0.0, 0.0, 50000)
        #     else:
        #         target = Pose(0.0, 0.0, 0.8)
        #         qcf.safe_position_setpoint(target)


        # # Move out and circle around Z axis
        # elif dt < 20:
        #     # print(f'[t={int(dt)}] Maneuvering - Circle around Z...')
        #     # Set target
        #     _x, _y = utils.pol2cart(circle_radius, phi)
        #     target = Pose(world.origin.x + _x,
        #                   world.origin.y + _y,
        #                   world.expanse)
        #     # Engage
        #     qcf.safe_position_setpoint(target)
        #
        # # Back to center
        # elif dt < 25:
        #     # print(f'[t={int(dt)}] Maneuvering - Center...')
        #     # Set target
        #     target = Pose(world.origin.x, world.origin.y, world.expanse)
        #     # Engage
        #     qcf.safe_position_setpoint(target)
        #
        # # Move out and circle around Y axis
        # elif dt < 40:
        #     # print(f'[t={int(dt)}] Maneuvering - Circle around Y...')
        #     # Set target
        #     _x, _z = utils.pol2cart(circle_radius, phi)
        #     target = Pose(world.origin.x + _x,
        #                   world.origin.y,
        #                   world.expanse + _z)
        #     # Engage
        #     qcf.safe_position_setpoint(target)
        #
        # # Back to center
        # elif dt < 45:
        #     # print(f'[t={int(dt)}] Maneuvering - Center...')
        #     # Set target
        #     target = Pose(world.origin.x, world.origin.y, world.expanse)
        #     # Engage
        #     qcf.safe_position_setpoint(target)
        #
        # # Move and circle around X axis
        # elif dt < 60:
        #     # print(f'[t={int(dt)}] Maneuvering - Circle around X...')
        #     # Set target
        #     _y, _z = utils.pol2cart(circle_radius, phi)
        #     target = Pose(world.origin.x,
        #                   world.origin.y + _y,
        #                   world.expanse + _z)
        #     # Engage
        #     qcf.safe_position_setpoint(target)
        #
        # # Back to center
        # elif dt < 65:
        #     # print(f'[t={int(dt)}] Maneuvering - Center...')
        #     # Set target
        #     target = Pose(world.origin.x, world.origin.y, world.expanse)
        #     # Engage
        #     qcf.safe_position_setpoint(target)

        else:
            break

    # Land
    while qcf.pose.z > 0.1:
        qcf.land_in_place()

    # Data logging close
    lg_stab.stop()
    print('Logging Finished.')

    # Joystick quit
    pygame.quit()
