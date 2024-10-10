import numpy as np
import matplotlib.pyplot as plt


def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def assign_targets_to_drones(drone_positions, target_positions):
    # Initialize paths for each drone
    drone_paths = {i: [drone_positions[i]] for i in range(len(drone_positions))}
    remaining_targets = target_positions.copy()

    while remaining_targets:
        # Find the nearest target for each drone
        for i in range(len(drone_positions)):
            if remaining_targets:
                current_position = drone_paths[i][-1]
                nearest_target = min(remaining_targets, key=lambda p: distance(current_position, p))
                drone_paths[i].append(nearest_target)
                remaining_targets.remove(nearest_target)

    return drone_paths


def assign_targets_with_additional_point(drones, targets, additional_target, assigned_drone_index, landing=None):
    # Initialize paths for each drone
    paths = {i: [drones[i]] for i in range(len(drones))}
    remaining_targets = targets.copy()

    # Assign the additional target to the specific drone
    paths[assigned_drone_index].append(additional_target)

    # Ensure the additional target is removed from the remaining list (if it's there)
    if additional_target in remaining_targets:
        remaining_targets.remove(additional_target)

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

    return paths


# Example usage
target_positions = [[2.0, -0.7], [-0.1, -0.9], [-2.0, -1.0], [-1.0, 0.7], [2.0, 1.0], [-0.5, -0.5]]
new_target_positions = [1.6, 1.1]
drones = [[0, 0], [3, 3]]

# Scenario 1: Assign the additional point to the 1st drone (index 0)
paths_scenario_1 = assign_targets_with_additional_point(drones, target_positions, new_target_positions, assigned_drone_index=0)

# Scenario 2: Assign the additional point to the 2nd drone (index 1)
paths_scenario_2 = assign_targets_with_additional_point(drones, target_positions, new_target_positions, assigned_drone_index=1)

print("Scenario 1 paths:", paths_scenario_1)
print("Scenario 2 paths:", paths_scenario_2)


# # Example usage
# drone_positions = [[0, 0], [0, 1]]
# target_positions = [[1, 2], [10, 6], [-1, 5], [6, 9], [9, 3]]
#
# # Calculate paths for each drone
# drone_paths = assign_targets_to_drones(drone_positions, target_positions)
#
# print(drone_paths[0])
# print(drone_paths[1])
#
# # Plot the paths
# plt.figure(figsize=(12, 8))
# colors = ['r', 'b']
# for i, path in drone_paths.items():
#     path_array = np.array(path)
#     plt.plot(path_array[:, 0], path_array[:, 1], f'{colors[i]}.-', label=f'Drone {i+1} Path')
#
# plt.scatter(np.array(target_positions)[:, 0], np.array(target_positions)[:, 1], c='g', label='Target Positions')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Drone Paths Using Nearest Neighbor Algorithm')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Print the paths
# for i, path in drone_paths.items():
#     print(f"Drone {i+1} Path: {path}")


# import threading
# import time
#
#
# def specific_function(arg1, arg2):
#     print(f"Function called with arguments {arg1} and {arg2} at:", time.time())
#
#
# def call_function_every_period(period, stop_event, *args):
#     if not stop_event.is_set():
#         specific_function(*args)
#         threading.Timer(period, call_function_every_period, [period, stop_event] + list(args)).start()
#
#
# # Example usage
# time_period = 0.1  # Set your desired time period here (in seconds)
# arg1 = "example_input1"
# arg2 = "example_input2"
#
# # Start time
# start_time = time.time()
#
# # Event to signal stopping the periodic function calls
# stop_event = threading.Event()
#
# # Start the periodic function calls
# call_function_every_period(time_period, stop_event, arg1, arg2)
#
# # Run your main program logic
# try:
#     while True:
#         dt = time.time() - start_time  # Calculate elapsed time
#         if dt > 10:  # Stop after 10 seconds
#             stop_event.set()
#             break
#
#         # Other code can go here
#
#         # Optional: Sleep to prevent high CPU usage in this loop
#         time.sleep(0.01)
# except KeyboardInterrupt:
#     stop_event.set()
#     print("Stopped by user")