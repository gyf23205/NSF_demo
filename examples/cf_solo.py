"""
qfly | Qualisys Drone SDK Example Script: Solo Crazyflie

Takes off, flies circles around Z, Y, X axes.
ESC to land at any time.
"""


import pynput
from time import sleep, time

from qfly import Pose, QualisysCrazyflie, World, utils

# SETTINGS
cf_body_name = 'nsf11'  # QTM rigid body name
cf_uri = 'radio://0/80/2M/E7E7E7E711'  # Crazyflie address
cf_marker_ids = [11, 12, 13, 14] # Active marker IDs
circle_radius = 0.5 # Radius of the circular flight path
circle_speed_factor = 0.12 # How fast the Crazyflie should move along circle

############
# print('parameter set')
# import cflib.crtp
# from cflib.crazyflie import Crazyflie
#
# cflib.crtp.init_drivers(enable_debug_driver=False)
# cf = Crazyflie(rw_cache='./cache')
# cf.open_link(cf_uri)
# cf.param.set_value('activeMarker.front', cf_marker_ids[0])
# cf.param.set_value('activeMarker.right', cf_marker_ids[1])
# cf.param.set_value('activeMarker.back', cf_marker_ids[2])
# cf.param.set_value('activeMarker.left', cf_marker_ids[3])
# cf.close_link()
# print('parameter set complete')
############

############
# Logging test
import logging
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
import csv
log_name = 'test_log.csv'
with open(log_name, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['time', 'x', 'y', 'z'])

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)


def log_stab_callback(timestamp, data, logconf):
    print('[%d][%s]: %s' % (timestamp, logconf.name, data))
    with open(log_name, 'a', newline='') as file_:
        writer_line = csv.writer(file_)
        writer_line.writerow([timestamp, data['kalman.stateX'], data['kalman.stateY'], data['kalman.stateZ']])


def simple_log_async(scf, logconf):
    cf = scf.cf
    cf.log.add_config(logconf)
    logconf.data_received_cb.add_callback(log_stab_callback)
    logconf.start()
    sleep(5)
    logconf.stop()
############


# Watch key presses with a global variable
last_key_pressed = None


# Set up keyboard callback
def on_press(key):
    """React to keyboard."""
    global last_key_pressed
    last_key_pressed = key
    if key == pynput.keyboard.Key.esc:
        fly = False


# Rotation matrix to quaternion
# def rotmatrix2quaternion(rotmatrix):
#     qw = utils.sqrt(1 + rotmatrix[0][0] + rotmatrix[1][1] + rotmatrix[2][2]) / 2
#     qx = utils.sqrt(1 + rotmatrix[0][0] - rotmatrix[1][1] - rotmatrix[2][2]) / 2
#     qy = utils.sqrt(1 - rotmatrix[0][0] + rotmatrix[1][1] - rotmatrix[2][2]) / 2
#     qz = utils.sqrt(1 - rotmatrix[0][0] - rotmatrix[1][1] + rotmatrix[2][2]) / 2
#     ql = utils.sqrt(qx ** 2 + qy ** 2 + qz ** 2 + qw ** 2)
#     return [qx / ql, qy / ql, qz / ql, qw / ql]


# Listen to the keyboard
listener = pynput.keyboard.Listener(on_press=on_press)
listener.start()


# Set up world - the World object comes with sane defaults
world = World(speed_limit=1.0)


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
    # Logging test
    lg_stab = LogConfig(name='Stabilizer', period_in_ms=500)
    lg_stab.add_variable('kalman.stateX', 'float')
    lg_stab.add_variable('kalman.stateY', 'float')
    lg_stab.add_variable('kalman.stateZ', 'float')

    # with SyncCrazyflie(cf_uri, cf=Crazyflie(rw_cache='./cache')) as scf:
    #     simple_log_async(scf, lg_stab)

    qcf.cf.log.add_config(lg_stab)
    lg_stab.data_received_cb.add_callback(log_stab_callback)
    lg_stab.start()
    sleep(1.0)
    ######################

    # MAIN LOOP WITH SAFETY CHECK
    while(qcf.is_safe()):
        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            break

        # Mind the clock
        dt = time() - t

        # Calculate Crazyflie's angular position in circle, based on time
        phi = circle_speed_factor * dt * 360

        # Take off and hover in the center of safe airspace for 5 seconds
        if dt < 5:
            # print(f'[t={int(dt)}] Maneuvering - Center...')
            # Set target
            target = Pose(world.origin.x, world.origin.y, 0.5 * world.expanse[2])
            # Engage
            qcf.safe_position_setpoint(target)

        # Move out and circle around Z axis
        elif dt < 20:
            # print(f'[t={int(dt)}] Maneuvering - Circle around Z...')
            # Set target
            _x, _y = utils.pol2cart(circle_radius, phi)
            target = Pose(world.origin.x + _x,
                          world.origin.y + _y,
                          0.5 * world.expanse[2])
            # Engage
            qcf.safe_position_setpoint(target)

        # Back to center
        elif dt < 25:
            # print(f'[t={int(dt)}] Maneuvering - Center...')
            # Set target
            target = Pose(world.origin.x, world.origin.y, 0.5 * world.expanse[2])
            # Engage
            qcf.safe_position_setpoint(target)

        # Move out and circle around Y axis
        elif dt < 40:
            # print(f'[t={int(dt)}] Maneuvering - Circle around Y...')
            # Set target
            _x, _z = utils.pol2cart(circle_radius, phi)
            target = Pose(world.origin.x + _x,
                          world.origin.y,
                          0.5 * world.expanse[2] + _z)
            # Engage
            qcf.safe_position_setpoint(target)

        # Back to center
        elif dt < 45:
            # print(f'[t={int(dt)}] Maneuvering - Center...')
            # Set target
            target = Pose(world.origin.x, world.origin.y, 0.5 * world.expanse[2])
            # Engage
            qcf.safe_position_setpoint(target)

        # Move and circle around X axis
        elif dt < 60:
            # print(f'[t={int(dt)}] Maneuvering - Circle around X...')
            # Set target
            _y, _z = utils.pol2cart(circle_radius, phi)
            target = Pose(world.origin.x,
                          world.origin.y + _y,
                          0.5 * world.expanse[2] + _z)
            # Engage
            qcf.safe_position_setpoint(target)

        # Back to center
        elif dt < 65:
            # print(f'[t={int(dt)}] Maneuvering - Center...')
            # Set target
            target = Pose(world.origin.x, world.origin.y, 0.5 * world.expanse[2])
            # Engage
            qcf.safe_position_setpoint(target)

        else:
            break

    # Land
    while (qcf.pose.z > 0.1):
        qcf.land_in_place()

    ######################
    # Data logging test
    lg_stab.stop()

    print('Logging Finished.')
    ######################
