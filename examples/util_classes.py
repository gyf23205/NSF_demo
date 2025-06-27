import pygame
import numpy as np
from constants import *


class Font:
    def __init__(self, font_name, size, pos):
        self.font = pygame.font.SysFont(font_name, size)
        self.font_name = font_name
        self.size = size
        self.pos = pos
        self.texts = []

    def update(self, content):
        text = self.font.render(content, True, BLACK)
        posx = self.pos[0]
        posy = self.pos[1] + len(self.texts) * self.size * line_height
        self.texts.append((text, (posx, posy)))

    def clear(self):
        self.texts = []


class Button:
    def __init__(self, rect, color, text, text_color=BLACK, font_name=FONT, font_size=FONT_SIZE):
        self.rect = pygame.Rect(rect)
        self.color = color
        self.text = text
        self.text_color = text_color
        self.font = pygame.font.SysFont(font_name, font_size)
        self.text_surf = self.font.render(text, True, text_color)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        surface.blit(self.text_surf, self.text_rect)

    def is_clicked(self, pos):
        if self.rect.collidepoint(pos):
            return True
        return False
    
    def handle_event(self, event):
        if event is not None and event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_clicked(event.pos):
                return True
        return False
    
class TextInput:
    def __init__(self, rect, color, maximum, text_color=BLACK, font_name=FONT, font_size=FONT_SIZE):
        self.rect = pygame.Rect(rect)
        self.color = color
        self.maximum = maximum  # Maximum number of characters
        self.text_color = text_color
        self.font = pygame.font.SysFont(font_name, font_size)
        self.text = ""
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Toggle active state if clicked
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False  # Optionally finish editing on Enter
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if int(self.text + event.unicode) <= self.maximum: 
                    self.text += event.unicode

    def draw(self, surface):
        # Draw the input box
        pygame.draw.rect(surface, self.color, self.rect, 2 if self.active else 1)
        # Render the text
        txt_surface = self.font.render(self.text, True, self.text_color)
        surface.blit(txt_surface, (self.rect.x+5, self.rect.y+5))


class Task:
    def __init__(self, surface, task_id, target_loc, task_pos, assigned_drone=None, assigned_gv=None):
        self.task_id = task_id
        self.surface = surface
        self.x0, self.y0 = task_pos
        self.target_pos = target_loc
        self.assigned_drone = assigned_drone if assigned_drone is not None else 'None'
        self.assigned_gv = assigned_gv  if assigned_gv is not None else 'None'

        self.grid_width = 80
        self.grid_height = line_height * FONT_SIZE
        self.task_id_text = Font(FONT, FONT_SIZE, (self.x0, self.y0))
        self.target_pos_text = Font(FONT, FONT_SIZE, (self.x0 + self.grid_width, self.y0))
        self.assigned_drone_input = TextInput((self.x0 + 2 * self.grid_width, self.y0,self.grid_width, self.grid_height), color=WHITE, maximum=n_drones)
        self.assigned_gv_input = TextInput((self.x0 + 3 * self.grid_width, self.y0, self.grid_width, self.grid_height), color=WHITE, maximum=n_gvs)

        self.task_id_text.update(f'     {self.task_id}')
        self.target_pos_text.update(f'{self.target_pos}')
        self.assigned_drone_input.text = '          ' + str(self.assigned_drone)
        self.assigned_gv_input.text = '                  ' + str(self.assigned_gv)

    # def update(self, task_pos):
    #     self.x0, self.y0 = task_pos
    #     self.task_id_text.pos = (self.x0, self.y0)
    #     self.target_pos_text.pos = (self.x0 + self.grid_width, self.y0)
    #     self.assigned_drone_input.rect.topleft = (self.x0 + 2 * self.grid_width, self.y0)
    #     self.assigned_gv_input.rect.topleft = (self.x0 + 3 * self.grid_width, self.y0)
        
    def draw(self):
        for text in self.task_id_text.texts:
            self.surface.blit(text[0], text[1])
        for text in self.target_pos_text.texts:
            self.surface.blit(text[0], text[1])
        self.assigned_drone_input.draw(self.surface)
        self.assigned_gv_input.draw(self.surface)

    def handle_event(self, event):
        self.assigned_drone_input.handle_event(event)
        self.assigned_gv_input.handle_event(event)