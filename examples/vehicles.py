import numpy as np
import pygame


class Vehicle:
    '''A class to graphically represent a drone in a simulation.'''
    def __init__(self, file_name, surface, sc=0.1, rt=0.0):
        self.surface = surface
        self.sc = sc
        self.altitude = 0.0
        self.size = None
        self.tick = 0
        self.rt = rt
        self.rect = None
        self.figure = None
        self.current_target = None
        self.load(file_name)

    def load(self, file_name):
        self.figure = pygame.image.load(file_name)
        self.scale()
        self.rotate(self.rt)
        self.rect = self.figure.get_rect()
        # self.rect.center = tuple(self.position)

    def update(self):
        # Adhoc data input: start
        # self.streaming.state_generator()
        # self.position[0] = self.streaming.position[0]
        # self.position[1] = self.streaming.position[1]
        # self.rt = self.streaming.rt
        # Adhoc data input: done
        # self.rect = self.figure.get_rect()
        self.tick += 1

    def scale(self):
        w = int(self.figure.get_width() * self.sc)
        h = int(self.figure.get_height() * self.sc)
        self.figure = pygame.transform.scale(self.figure, (w, h))
        # self.rect = self.figure.get_rect()

    def rotate(self, rt):
        self.figure = pygame.transform.rotate(self.figure, rt)
        # self.rect = self.figure.get_rect()

    def draw(self, center):
        self.rect.center = center
        self.surface.blit(self.figure, self.rect)



class VirtualDrone:
    '''A class to represent the physical states of a drone'''
    def __init__(self, idx, pos):
        self.idx = idx
        self.position = [pos[0], pos[1], 0]
        self.rt = 0
        self.health = 100.0

    # def set_position(self, target_):
    #     self.position = target_

    def takeoff_in_place(self, altitude):
        self.position[2] = min([self.position[2] + 0.01, altitude])

    def land_in_place(self):
        self.position[2] = max([self.position[2] - 0.01, 0.0])


class VirtualGV:
    '''A class to represent the physical states of a drone'''
    def __init__(self, idx, pos):
        self.idx = idx
        self.position = [pos[0], pos[1]]
        self.current_target = None
        self.carrying = False
        self.health = 100.0
        # self.rt = 0
        # self.health = 100.0

    # def set_position(self, target_):
    #     self.position = target_

    # def takeoff_in_place(self, altitude):
    #     self.position[2] = min([self.position[2] + 0.01, altitude])

    # def land_in_place(self):
    #     self.position[2] = max([self.position[2] - 0.01, 0.0])
