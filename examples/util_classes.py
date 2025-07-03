import pygame
import numpy as np
from constants import *


class Bar:
    def __init__(self, screen, rect, color=GREEN, maximum=100.0):
        self.screen = screen
        self.color = color
        self.maximum = maximum
        self.rect = pygame.Rect(rect)

    def draw(self, val):
        # Set val in [0, 100]
        val = max(0, min(100, val))
        if val >= 60:
            self.color = GREEN
        elif 30 <= val < 60:
            self.color = YELLOW
        elif val < 30:
            self.color = RED
        else:
            print("Invalid val number!")
        fill_rect = pygame.Rect((*self.rect.topleft, (val/100) * self.rect[2], self.rect[3]))
        pygame.draw.rect(self.screen, GREY, self.rect)
        pygame.draw.rect(self.screen, self.color,fill_rect)
        pygame.draw.rect(self.screen, BLACK, self.rect, 2)

class Font:
    def __init__(self, font_name, size, pos):
        self.font = pygame.font.SysFont(font_name, size)
        self.font_name = font_name
        self.size = size
        self.pos = pos
        self.texts = []
        self.rect = pygame.Rect((pos[0], pos[1], 0, 0))  # Initialize rect with position

    def update(self, content):
        text = self.font.render(self.set_digits(content), True, BLACK)
        posx = self.pos[0]
        posy = self.pos[1] + len(self.texts) * int(self.size * line_height)  # Stack text vertically
        self.texts.append((text, (posx, posy)))
        self.rect = pygame.Rect((posx, posy, text.get_width(), text.get_height()))  # Update rect size based on text

    def clear(self):
        self.texts = []

    def set_digits(self, content):
        if isinstance(content, str) and content[0] == '[' and content[-1] == ']':
            d1 = content[1:content.index(',')]
            d2 = content[content.index(',')+1:-1]
            content = '[{:.2f}, {:.2f}]'.format(float(d1), float(d2))
        return content



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
        self.lock = False  # Lock to prevent multiple activations
        self.temp_flag = False  # Temporary flag to track active state changes

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and not self.lock:
            # Toggle active state if clicked
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
            else:
                self.active = False
            self.lock = True
        
        if self.temp_flag != self.active:
            print(f"TextInput active state changed: {self.active}")
            print(f"TextInput text updated: {self.text}")
            self.temp_flag = self.active

        if event.type == pygame.MOUSEBUTTONUP and self.lock:
            # Reset lock on mouse button release
           self.lock = False

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False  # Optionally finish editing on Enter
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                print(f"TextInput text after backspace: {self.text}")
            else:
                # Check if the input is a digit and within the maximum limit
                if event.unicode.isdigit():
                    if int(self.text + event.unicode) <= self.maximum: 
                        self.text += event.unicode
                        print(f"TextInput text updated: {self.text}")
                    else:
                        print(f"TextInput text '{self.text + event.unicode}' exceeds maximum {self.maximum-1}")
                else:
                    print(f"TextInput received non-digit input: {event.unicode}")

    def draw(self, surface):
        # Fill the input box background
        pygame.draw.rect(surface, self.color, self.rect)  # Fill the rect
        # Draw the rim (border)
        pygame.draw.rect(surface, BLACK, self.rect, 5 if self.active else 1)
        # Render the text   
        txt_surface = self.font.render(self.text, True, self.text_color)
        surface.blit(txt_surface, (self.rect.x+5, self.rect.y+5))


