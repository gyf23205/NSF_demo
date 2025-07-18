"""
RRT_CONNECT_2D
@author: huiming zhou
"""

import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import euclidean

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

from rrt_2D import env, plotting
from rrt_2D import utils_rrt as utils


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class RrtConnect:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max):
        self.path = None    # Added by SY
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.s_goal, self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.V1, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new)

                if node_new_prim and not self.utils.is_collision(node_new_prim, node_near_prim):
                    self.V2.append(node_new_prim)

                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new)
                        if node_new_prim2 and not self.utils.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                if self.is_node_same(node_new_prim, node_new):
                    self.path = self.extract_path(node_new, node_new_prim)
                    # Set right direction of path: AD-HOC! Should be revised!
                    if np.linalg.norm(np.array(self.path[-1]) - np.array([self.s_goal.x, self.s_goal.y])) < 1e-2:
                        self.path.reverse()
                        # print('Path reversed for correction.')
                    return self.path

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid

        return None

    def smoothing(self):
        if not self.path:
            return []

        # Skip smoothing for short paths or stay-still
        if np.linalg.norm(np.array(self.path[0]) - np.array(self.path[-1])) < 1e-7:
            self.path = [self.path[0], self.path[0], self.path[0], self.path[0], self.path[0]]
            return []
        if len(self.path) < 7:
            return []

        # 1) Eliminate redundant waypoints
        # Initialize
        non_redundant_path = [self.path[0]]

        # Iterate through the path in reversed order
        for i in range(1, len(self.path)):
            if self.utils.is_collision(Node(non_redundant_path[-1]), Node(self.path[i])):
                non_redundant_path.append(self.path[i-1])

        # Add the last point in the path
        non_redundant_path.append(self.path[-1])

        # 2) Generate smooth trajectory using B-spline
        path = np.array(non_redundant_path)

        # Desired distance between points
        dist = 0.05

        # Fit a B-spline
        tck, u = splprep([path[:, 0], path[:, 1]], k=1, s=0)

        # Compute the total arc length of the B-spline
        num_points = 1000
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine = splev(u_fine, tck)
        points = np.vstack((x_fine, y_fine)).T

        # Sample points along the B-spline
        sampled_points = [points[0]]
        accumulated_dist = 0
        for i in range(1, len(points)):
            segment_dist = euclidean(points[i - 1], points[i])
            accumulated_dist += segment_dist
            if accumulated_dist >= dist:
                sampled_points.append(points[i])
                accumulated_dist = 0

        sampled_points.append(points[-1])
        sampled_points = np.array(sampled_points)

        # Ensure the trajectory starts and ends at the exact positions
        sampled_points[0] = path[0]
        sampled_points[-1] = path[-1]

        # Return (np.array not list)
        self.path = sampled_points

        # 3) Plotting, if necessary
        # self.plotting.animation_connect(self.V1, self.V2, self.path, "RRT_CONNECT")

    def add_wind_area(self, wind_area):
        self.env.added_circle = wind_area

    @staticmethod
    def change_node(node_new_prim, node_new_prim2):
        node_new = Node((node_new_prim2.x, node_new_prim2.y))
        node_new.parent = node_new_prim

        return node_new

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        if node_new_prim.x == node_new.x and \
                node_new_prim.y == node_new.y:
            return True

        return False

    def generate_random_node(self, sample_goal, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return sample_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    @staticmethod
    def extract_path(node_new, node_new_prim):
        path1 = [(node_new.x, node_new.y)]
        node_now = node_new

        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append((node_now.x, node_now.y))

        path2 = [(node_new_prim.x, node_new_prim.y)]
        node_now = node_new_prim

        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append((node_now.x, node_now.y))

        return list(list(reversed(path1)) + path2)

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    # x_start = (2, 2)  # Starting node
    # x_goal = (49, 24)  # Goal node
    x_start = (-2.0, -1.0)  # Starting node
    x_goal = (1.8, 0.9)  # Goal node

    # rrt_conn = RrtConnect(x_start, x_goal, 0.8, 0.05, 5000)
    rrt_conn = RrtConnect(x_start, x_goal, 0.08, 0.05, 5000)
    path = rrt_conn.planning()

    rrt_conn.plotting.animation_connect(rrt_conn.V1, rrt_conn.V2, path, "RRT_CONNECT")


if __name__ == '__main__':
    main()
