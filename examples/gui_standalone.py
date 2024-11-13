"""
Standalone GUI testbed for HAT research
Sooyung Byeon
10/03/2024

To do list:
- Fault actions: {sensor, actuator, communication...}
- Environment change actions: {wind, fog, other drones...}
- Multiple scenarios with different function allocation
- Function allocation change
- Mission update actions
- Data streaming and workload inference in real-time with HR/Cam
- Asking survey questions after performance
- Collision avoidance by altitude control
- Set search area - click targets and hit enter (Is this good enough?)

- Make classes
- Identify 'GUI parts' clearly
- Dual Windows (if necessary)
"""

import pynput
import numpy as np
from time import time, sleep
from qfly import World
from rrt_2D import rrt_connect
import game

# [Temporary] Function allocation
fa = 3  # {1: monitor + confirm, 2: + re-planning, 3: + fault}

# Drone Setting: Physical constraints
hover_duration = 10
stay_duration = 5
stay_distance = 0.1
separation_distance = 0.5
total_duration = 120
speed_constant = 5.0
indexing_time = [hover_duration, hover_duration]

# Set up the world
world = World()

# Watch key presses with a global variable
last_key_pressed = None


# Task allocation functions: should be modulated at some point
def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def assign_targets_to_drones(drones, targets, landing=None):
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

    if landing is not None:
        for i in range(len(landing)):
            paths[i].append(landing[i])

    return paths


def assign_targets_with_path_lengths(drones, targets, assigned_drone_index, landing=None):
    # Initialize paths for each drone
    paths = {i: [drones[i]] for i in range(len(drones))}
    remaining_targets = targets.copy()

    # Assign the additional target to the specific drone
    additional_target = remaining_targets.pop(assigned_drone_index)
    paths[assigned_drone_index].append(additional_target)

    while remaining_targets:
        # Find the nearest target for each drone
        for i in range(len(drones)):
            if remaining_targets:
                current_position = paths[i][-1]
                nearest_target = min(remaining_targets, key=lambda p: distance(current_position, p))
                paths[i].append(nearest_target)
                remaining_targets.remove(nearest_target)

    # Add landing positions if provided
    if landing is not None:
        for i in range(len(landing)):
            paths[i].append(landing[i])

    # Compute path lengths for each drone
    path_lengths = []
    for i in range(len(drones)):
        path_length = sum(distance(paths[i][j], paths[i][j+1]) for j in range(len(paths[i])-1))
        path_lengths.append(path_length)

    # Output 0 is total path length (output[0] = output[1] + output[2])
    output = [sum(path_lengths), path_lengths[0], path_lengths[1]]

    return output, paths


# Adhoc altitude control for collision avoidance
def avoid_altitude_control(drones, proximity=0.5):
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


# Make a virtual drone class at here
class VirtualDrone:
    def __init__(self):
        self.position = [0, 0, 0]
        self.rt = 0

    def set_position(self, target_):
        self.position = target_

    def takeoff_in_place(self, altitude):
        self.position[2] = min([self.position[2] + 0.01, altitude])

    def land_in_place(self):
        self.position[2] = max([self.position[2] - 0.01, 0.0])


drones = [VirtualDrone(), VirtualDrone()]
drones[0].position[0] = -1.2
drones[0].position[1] = -0.5
drones[1].position[0] = -1.2
drones[1].position[1] = 0.5

"""with ParallelContexts(*_qcfs) as qcfs:"""

# "fly" variable used for landing on demand
fly = True

# Log: omitted

# Task allocation
target_positions = [[2.0, -0.7], [-0.1, -0.9], [-2.0, -1.0], [-1.0, 0.7], [2.0, 1.0], [-0.5, -0.5]]
takeoff_positions = [[drones[0].position[0], drones[0].position[1]], [drones[1].position[0], drones[1].position[1]]]
drone_paths = assign_targets_to_drones(takeoff_positions, target_positions, landing=takeoff_positions)

# Remaining paths for re-planning
target_remaining = target_positions.copy()
new_target = []
new_target_positions = []
path1 = []
path2 = []

# RRT-connect for all drone paths
drone_trajectory = [[], []]
for d_inx in range(2):
    for p_idx in range(len(drone_paths[d_inx]) - 1):
        rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx], drone_paths[d_inx][p_idx + 1], 0.08, 0.05, 5000)
        rrt_conn.planning()
        rrt_conn.smoothing()
        drone_trajectory[d_inx].append(rrt_conn.path)

"""Initialization of main loop"""
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

# Target and takeoff positions for GUI: this part should be a function
target_gui = np.array(target_positions)
for k in range(len(target_positions)):
    target_gui[k][0] = 240.0 * target_gui[k][0] + 600
    target_gui[k][1] = -240.0 * target_gui[k][1] + 360
takeoff_gui = np.array(takeoff_positions)
for k in range(len(takeoff_positions)):
    takeoff_gui[k][0] = 240.0 * takeoff_gui[k][0] + 600
    takeoff_gui[k][1] = -240.0 * takeoff_gui[k][1] + 360
game_mgr.set_target(target=target_gui)
game_mgr.set_takeoff_positions(takeoff_gui)

# Time
t = time()
dt = 0
dt_prev = [hover_duration, hover_duration]

# Path
path_index = [0, 0]

"""Main Loop"""
while fly:

    # Land with ESC
    if last_key_pressed == pynput.keyboard.Key.esc:
        print(f'[t={int(dt)}] Escape by keyboard.')
        break

    # GUI input
    mode = game_mgr.input()

    # Mind the clock
    dt = time() - t

    # Mission completed
    if all(mission_complete):
        print(f'[t={int(dt)}] All missions completed!')
        break

    for idx in range(2):
        # GUI: this part should be a function
        game_mgr.objects[idx].position[0] = drones[idx].position[0] * 240.0 + 600
        game_mgr.objects[idx].position[1] = -drones[idx].position[1] * 240.0 + 360
        drones[idx].rt = np.random.normal(0, 0.01, 1)[0]
        game_mgr.objects[idx].rt = drones[idx].rt * 180 / np.pi
        game_mgr.objects[idx + 2].position[1] = -75.0 * drones[idx].position[2] + 200.0 + np.random.normal(0, 0.3, 1)[0]

        # Initial hover
        if dt < hover_duration:
            # Engage
            drones[idx].takeoff_in_place(0.5 * world.expanse[2])
            sleep(0.01)

        # Proceed target acquisition
        elif dt < total_duration:
            # Move to the current target
            target_current = drone_paths[idx][target_index[idx] + 1]
            # Mission completed?
            if mission_complete[idx]:
                drones[idx].land_in_place()
                sleep(0.01)
                continue
            else:
                # Mutual avoidance by altitude control: TBD
                # Temporary
                current_trajectory = drone_trajectory[idx][target_index[idx]]
                # Find the target positions: {make indexing as a function for better interpretability}
                increment = speed_constant * (dt - dt_prev[idx])
                path_index[idx] += increment
                picked = min(int(path_index[idx]) + 1, len(current_trajectory))
                target = [current_trajectory[-picked][0], current_trajectory[-picked][1],
                          0.5 * world.expanse[2]]
                drones[idx].set_position(target)
                # Time
                dt_prev[idx] = dt
                sleep(0.01)

            # Check distance to the target
            target_distance[idx] = distance(target_current, drones[idx].position[0:2])

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
                        # To record response time
                        game_mgr.victim_timing[idx] = dt

                    # Once victim is selected, close it: only if there is no unassigned target
                    if game_mgr.target_decided:
                        if game_mgr.victim_clicked[idx]:
                            game_mgr.victim_id[idx] = 0
                            game_mgr.victim_detected[idx] = False
                            # To record response time
                            game_mgr.missions[idx].response_time.append(dt - game_mgr.victim_timing[idx])
                            game_mgr.victim_timing[idx] = 0
                            print(f'Response time by drone {int(idx + 1)}: {game_mgr.missions[idx].response_time}')
                            # Print status
                            print(f'[t={int(dt)}] Drone {int(idx + 1)}: target {int(target_index[idx] + 1)} accomplished')
                            target_remaining.remove(target_current)
                            # Update and reset
                            target_index[idx] += 1
                            stay_flag[idx] = True
                            stay_time[idx] = 0
                            # Temporary
                            indexing_time[idx] = dt
                            path_index[idx] = 0
                    else:
                        game_mgr.victim_block_choice[idx] = True
                else:
                    mission_complete[idx] = True

            # Mission (target) update: triggered by the number of remaining targets
            if idx == 0:    # Do only once: 1st drone
                if len(target_remaining) == 4 and dt > 18.0 and not game_mgr.new_target_triggered:
                    # New target detected (comm from mission control)
                    print(f'[t={int(dt)}] New target identified')
                    new_target = [1.6, 1.1]
                    new_target_positions.append(new_target)
                    new_target_gui = np.array(new_target_positions)
                    new_target_gui[0][0] = 240 * new_target_positions[0][0] + 600.0
                    new_target_gui[0][1] = -240 * new_target_positions[0][1] + 360.0
                    game_mgr.set_target(new_target=new_target_gui)
                    # Update remaining
                    target_remaining.append(new_target)
                    # Close the case
                    game_mgr.new_target_triggered = True
                    game_mgr.target_decided = False

                    # Compute distance with additional target
                    new_start_positions = [drone_paths[0][target_index[0] + 1], drone_paths[1][target_index[1] + 1]]
                    # [Temporary: code B]
                    target_remaining_copy = target_remaining.copy()
                    target_remaining_copy.remove(new_start_positions[0])
                    target_remaining_copy.remove(new_start_positions[1])
                    # Two re-planning scenarios
                    output1, path1 = assign_targets_with_path_lengths(new_start_positions, target_remaining_copy, 0,
                                                                      landing=takeoff_positions)
                    output2, path2 = assign_targets_with_path_lengths(new_start_positions, target_remaining_copy, 1,
                                                                      landing=takeoff_positions)
                    game_mgr.planning_distances = output1 + output2

                # Decision-making on the new target (triggered) by mouse action
                # [Temporary] Function allocation
                if fa == 1:
                    game_mgr.target_clicked = 2
                if game_mgr.new_target_triggered and not game_mgr.target_decided and game_mgr.target_clicked:
                    print(f'[t={int(dt)}] Decision made on the new target')
                    # Restore block if there is any
                    game_mgr.victim_block_choice = [False, False]
                    # Target decision made
                    game_mgr.target_decided = True
                    # Reset the planning distance to zeros
                    game_mgr.planning_distances = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    # Selecting one re-planning scenario
                    if game_mgr.target_clicked == 1:
                        drone_paths = path1
                    elif game_mgr.target_clicked == 2:
                        drone_paths = path2
                    new_drone_trajectory = [[], []]
                    for d_inx in range(2):
                        for p_idx in range(len(drone_paths[d_inx]) - 1):
                            rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx], drone_paths[d_inx][p_idx + 1],
                                                              0.08, 0.05, 5000)
                            rrt_conn.planning()
                            rrt_conn.smoothing()
                            new_drone_trajectory[d_inx].append(rrt_conn.path)
                    # Maintain the current trajectory
                    drone_trajectory = [[drone_trajectory[0][target_index[0]]], [drone_trajectory[1][target_index[1]]]]
                    for i in range(2):
                        for j in range(len(new_drone_trajectory[i])):
                            drone_trajectory[i].append(new_drone_trajectory[i][j])
                    target_index = [0, 0]
                    # [Temporary] Recover the overlapped start positions (target_current)
                    drone_paths[0].insert(0, new_start_positions[0])
                    drone_paths[1].insert(0, new_start_positions[1])

            # Wind (fault) condition
            if idx == 0:
                # Trigger windy condition based on time
                if not game_mgr.wind_triggered and dt > 30.0:
                    game_mgr.wind_danger = True
                    game_mgr.wind_triggered = True
                    game_mgr.wind_decided = False
                    print(f'[t={int(dt)}] Environmental change: dangerous wind')
                    speed_constant = 2.0
                    # Put obstacle avoidance here
                    wind = [0, 0, 0.5]
                    wind_gui = wind.copy()
                    wind_gui[0] = 240 * wind[0] + 600.0
                    wind_gui[1] = -240 * wind[1] + 360.0
                    wind_gui[2] = 240 * wind[2]
                    game_mgr.set_wind(wind_gui)

                # Turn-off windy condition based on time
                if game_mgr.wind_danger and game_mgr.wind_triggered and dt > 70.0:
                    game_mgr.wind_danger = False
                    game_mgr.wind_decided = True
                    print(f'[t={int(dt)}] Environmental change: stable wind')
                    # Remove wind graphic
                    game_mgr.reset_wind()
                    # Speed adjustment
                    speed_constant = 5.0
                    # Task allocation
                    current_start = [[drones[0].position[0], drones[0].position[1]],
                                     [drones[1].position[0], drones[1].position[1]]]
                    drone_paths = assign_targets_to_drones(current_start, target_remaining,
                                                           landing=takeoff_positions)
                    # RRT-connect for all drone paths
                    drone_trajectory = [[], []]
                    for d_inx in range(2):
                        for p_idx in range(len(drone_paths[d_inx]) - 1):
                            rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx],
                                                              drone_paths[d_inx][p_idx + 1], 0.08, 0.05, 5000)
                            rrt_conn.utils.update_obs([], [], [])
                            rrt_conn.planning()
                            rrt_conn.smoothing()
                            drone_trajectory[d_inx].append(rrt_conn.path)
                    target_index = [0, 0]
                    # Time correction
                    path_index = [0, 0]
                    dt_prev[0] = dt
                    dt_prev[1] = dt

                # [Temporary] Function allocation
                # If fa = 1 or 2, skip the wind change response
                if fa == 1 or fa == 2:
                    game_mgr.wind_clicked = 1
                # Ask for decision
                # If fa = 3, ask for human's decision
                if game_mgr.wind_danger and game_mgr.wind_triggered and not game_mgr.wind_decided:
                    if game_mgr.wind_clicked == 1:
                        game_mgr.wind_decided = True
                        print(f'[t={int(dt)}] Change routes')
                        # Speed adjustment
                        speed_constant = 5.0
                        # Task allocation
                        current_start = [[drones[0].position[0], drones[0].position[1]],
                                         [drones[1].position[0], drones[1].position[1]]]
                        drone_paths = assign_targets_to_drones(current_start, target_remaining,
                                                               landing=takeoff_positions)
                        # RRT-connect for all drone paths
                        drone_trajectory = [[], []]
                        for d_inx in range(2):
                            for p_idx in range(len(drone_paths[d_inx]) - 1):
                                rrt_conn = rrt_connect.RrtConnect(drone_paths[d_inx][p_idx],
                                                                  drone_paths[d_inx][p_idx + 1], 0.08, 0.05, 5000)
                                rrt_conn.utils.update_obs([wind], [], [])
                                rrt_conn.planning()
                                rrt_conn.smoothing()
                                drone_trajectory[d_inx].append(rrt_conn.path)
                        target_index = [0, 0]
                        # Time correction
                        path_index = [0, 0]
                        dt_prev[0] = dt
                        dt_prev[1] = dt

                    elif game_mgr.wind_clicked == 2:
                        game_mgr.wind_decided = True
                        print(f'[t={int(dt)}] Maintain routes')

        else:
            fly = False

    # GUI rendering
    game_mgr.update()
    game_mgr.render()

# Land till the end
while max(drones[0].position[2], drones[1].position[2]) > 0.01:
    for idx in range(2):
        drones[idx].land_in_place()
        sleep(0.01)
        # This should be a function
        game_mgr.objects[idx].position[0] = drones[idx].position[0] * 240.0 + 600
        game_mgr.objects[idx].position[1] = -drones[idx].position[1] * 240.0 + 360
        drones[idx].rt = np.random.normal(0, 0.01, 1)[0]
        game_mgr.objects[idx].rt = drones[idx].rt * 180 / np.pi
        game_mgr.objects[idx + 2].position[1] = -75.0 * drones[idx].position[2] + 200.0 + np.random.normal(0, 0.3, 1)[0]
    # GUI rendering
    game_mgr.update()
    game_mgr.render()

print('End GUI standalone mode.')
