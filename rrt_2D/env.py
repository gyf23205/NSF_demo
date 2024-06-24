"""
Environment for rrt_2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        # self.x_range = (0, 50)
        # self.y_range = (0, 30)
        self.x_range = (-2.5, 2.5)
        self.y_range = (-1.5, 1.5)
        self.obs_boundary = self.obs_boundary()
        self.obs_circle = self.obs_circle()
        self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        # obs_boundary = [
        #     [0, 0, 1, 30],
        #     [0, 30, 50, 1],
        #     [1, 0, 50, 1],
        #     [50, 1, 1, 30]
        # ]
        obs_boundary = [
            [-2.5, -1.5, 0.1, 3],
            [-2.5, 1.5, 5, 0.1],
            [-2.5 + 0.1, -1.5, 5, 0.1],
            [2.5, -1.5 + 0.1, 0.1, 3]
        ]
        # [start_x, start_y, horizontal length, vertical length]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        # obs_rectangle = [
        #     [14, 12, 8, 2],
        #     [18, 22, 8, 3],
        #     [26, 7, 2, 12],
        #     [32, 14, 10, 2]
        # ]
        obs_rectangle = [
            [1.4 - 2.5, 1.2 - 1.5, 0.8, 0.2],
            [1.8 - 2.5, 2.2 - 1.5, 0.8, 0.3],
            [2.6 - 2.5, 0.7 - 1.5, 0.2, 1.2],
            [3.2 - 2.5, 1.4 - 1.5, 1.0, 0.2]
        ]
        # [start_x, start_y, horizontal length, vertical length]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        # obs_cir = [
        #     [7, 12, 3],
        #     [46, 20, 2],
        #     [15, 5, 2],
        #     [37, 7, 3],
        #     [37, 23, 3]
        # ]
        obs_cir = [
            [0.7 - 2.5, 1.2 - 1.5, 0.3],
            [4.6 - 2.5, 2.0 - 1.5, 0.2],
            [1.5 - 2.5, 0.5 - 1.5, 0.2],
            [3.7 - 2.5, 0.7 - 1.5, 0.3],
            [3.7 - 2.5, 2.3 - 1.5, 0.3]
        ]
        # [center_x, center_y, radius]
        return obs_cir
