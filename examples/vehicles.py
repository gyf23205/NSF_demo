import numpy as np
import pygame
from constants import *


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
        self.original_w = None
        self.current_target = None
        self.load(file_name)

    def load(self, file_name):
        self.figure = pygame.image.load(file_name)
        self.original_w = self.figure.get_width()
        self.original_h = self.figure.get_height()
        self.scale()
        self.rotate(self.rt)
        self.rect = self.figure.get_rect()

    def update(self):
        self.tick += 1

    def scale(self):
        w = int(self.original_w * self.sc)
        h = int(self.original_h * self.sc)
        self.figure = pygame.transform.scale(self.figure, (w, h))
        # self.rect = self.figure.get_rect()

    def rotate(self, rt):
        self.figure = pygame.transform.rotate(self.figure, rt)
        # self.rect = self.figure.get_rect()

    def draw(self, center, altitude=0.7):
        self.altitude = altitude
        # self.scale()
        self.rect.center = center
        self.surface.blit(self.figure, self.rect)



class VirtualDrone:
    '''A class to represent the physical states of a drone'''
    def __init__(self, idx, pos):
        self.idx = idx
        self.position = [pos[0], pos[1], 0]
        self.rt = 0
        self.health = 100.0
        self.status = 'idle'
        self.inspecting_alt = (50.0 / 100.0) * 2

    def set_position(self, target_):
        self.position = target_

    def takeoff_in_place(self, altitude):
        self.position[2] = min([self.position[2] + 0.01, altitude])

    def land_in_place(self):
        self.position[2] = max([self.position[2] - 0.01, 0.0])

    def down4inspect(self):
        self.position[2] = max([self.position[2] - 0.01, self.inspecting_alt])
    
    def up4move(self):
        self.position[2] = min([self.position[2] + 0.01, max_altitude])



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
