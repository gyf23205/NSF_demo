"""
qfly | Qualisys Drone SDK Example Script: Interactive Crazyflie with Deck

The drone flies along the YZ plane while centered at 0 along the X plane.
The Y and Z coordinates track another Crazyflie body equipped with an
Active Marker Deck.
ESC to land at any time.
"""

import pynput
from time import sleep

from qfly import ParallelContexts, Pose, QualisysCrazyflie, QualisysDeck, World

# SETTINGS
cf_body_names = [
    'nsf1',
    'nsf11'
]
# Addresses
cf_uris = [
    'radio://0/80/2M/E7E7E7E711',
    'radio://0/80/2M/E7E7E7E731'
]
# Marker ids
cf_marker_ids = [
    [1, 2, 3, 4],
    [11, 12, 13, 14]
]
# Deck
deck_body_name = 'nsf2'
deck_uri = 'radio://0/81/2M/E7E7E7E740'
deck_marker_ids = [5, 6, 7, 8]

# Watch key presses with a global variable
last_key_pressed = None


# Set up keyboard callback
def on_press(key):
    """React to keyboard."""
    global last_key_pressed
    last_key_pressed = key
    if key == pynput.keyboard.Key.esc:
        fly = False


# Listen to the keyboard
listener = pynput.keyboard.Listener(on_press=on_press)
listener.start()

# Set up world - the World object comes with sane defaults
world = World()

# Stack up context managers
qcfs = [QualisysCrazyflie(cf_body_name,
                          cf_uri,
                          world,
                          marker_ids=cf_marker_id,
                          qtm_ip="192.168.123.2")
        for cf_body_name, cf_uri, cf_marker_id
        in zip(cf_body_names, cf_uris, cf_marker_ids)]

deck = QualisysDeck(deck_body_name,
                    deck_uri,
                    deck_marker_ids,
                    qtm_ip="192.168.123.2")

with ParallelContexts(*qcfs, deck):
    print("Beginning maneuvers...")

    # MAIN LOOP WITH SAFETY CHECK
    while qcfs[0].is_safe():

        # Terminate upon Esc command
        if last_key_pressed == pynput.keyboard.Key.esc:
            break
        # Take off and hover in the center of safe airspace for 5 seconds
        # Set target
        x = world.origin.x
        y = world.origin.y
        z = 1.0
        if deck.pose is not None:
            y = deck.pose.y
            z = deck.pose.z
        target1 = Pose(x, y, z)
        target2 = Pose(x - 1.0, y, z)
        # Engage
        qcfs[0].safe_position_setpoint(target1)
        qcfs[1].safe_position_setpoint(target2)
        sleep(0.02)
        continue

    # Land calmly
    qcfs[0].land_in_place()
