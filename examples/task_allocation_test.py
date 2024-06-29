import threading
import time


def specific_function(arg1, arg2):
    print(f"Function called with arguments {arg1} and {arg2} at:", time.time())


def call_function_every_period(period, stop_event, *args):
    if not stop_event.is_set():
        specific_function(*args)
        threading.Timer(period, call_function_every_period, [period, stop_event] + list(args)).start()


# Example usage
time_period = 0.1  # Set your desired time period here (in seconds)
arg1 = "example_input1"
arg2 = "example_input2"

# Start time
start_time = time.time()

# Event to signal stopping the periodic function calls
stop_event = threading.Event()

# Start the periodic function calls
call_function_every_period(time_period, stop_event, arg1, arg2)

# Run your main program logic
try:
    while True:
        dt = time.time() - start_time  # Calculate elapsed time
        if dt > 10:  # Stop after 10 seconds
            stop_event.set()
            break

        # Other code can go here

        # Optional: Sleep to prevent high CPU usage in this loop
        time.sleep(0.01)
except KeyboardInterrupt:
    stop_event.set()
    print("Stopped by user")

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def distance(point1, point2):
#     return np.linalg.norm(np.array(point1) - np.array(point2))
#
#
# def assign_targets_to_drones(drone_positions, target_positions):
#     # Initialize paths for each drone
#     drone_paths = {i: [drone_positions[i]] for i in range(len(drone_positions))}
#     remaining_targets = target_positions.copy()
#
#     while remaining_targets:
#         # Find the nearest target for each drone
#         for i in range(len(drone_positions)):
#             if remaining_targets:
#                 current_position = drone_paths[i][-1]
#                 nearest_target = min(remaining_targets, key=lambda p: distance(current_position, p))
#                 drone_paths[i].append(nearest_target)
#                 remaining_targets.remove(nearest_target)
#
#     return drone_paths
#
#
# # Example usage
# drone_positions = [[0, 0], [0, 1]]
# target_positions = [[1, 2], [10, 6], [-1, 5], [6, 9], [9, 3]]
#
# # Calculate paths for each drone
# drone_paths = assign_targets_to_drones(drone_positions, target_positions)
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
