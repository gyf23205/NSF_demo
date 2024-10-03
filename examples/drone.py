import numpy as np
import pygame


class Drone:
    def __init__(self, file_name, pos_x=750, pos_y=450, sc=0.1, rt=0.0):
        self.position = [pos_x, pos_y]
        self.sc = sc
        self.rect = []
        self.tick = 0
        self.rt = rt
        self.figure = []
        self.rect = []
        self.surface = None
        self.load(file_name)

    def load(self, file_name):
        self.figure = pygame.image.load(file_name)
        self.surface = self.figure
        self.scale()
        self.rotate(self.rt)
        self.rect = self.surface.get_rect()
        self.rect.center = tuple(self.position)

    def update(self):
        # Adhoc data input: start
        # self.streaming.state_generator()
        # self.position[0] = self.streaming.position[0]
        # self.position[1] = self.streaming.position[1]
        # self.rt = self.streaming.rt
        # Adhoc data input: done
        self.rect = self.surface.get_rect()
        self.tick += 1

    def scale(self):
        w = int(self.surface.get_width() * self.sc)
        h = int(self.surface.get_height() * self.sc)
        self.surface = pygame.transform.scale(self.surface, (w, h))
        self.rect = self.surface.get_rect()

    def rotate(self, rt):
        self.surface = pygame.transform.rotate(self.surface, rt)
        self.rect = self.surface.get_rect()
