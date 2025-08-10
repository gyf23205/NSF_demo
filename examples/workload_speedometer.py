import pygame
import numpy as np
import time

class WorkloadSpeedometer:
    def __init__(self, x, y, radius=60, smoothing_speed=1):
        self.center = (x, y)
        self.radius = radius
        self._target = 0.0
        self._display = 0.0
        self._last_time = time.time()
        self._smoothing_speed = smoothing_speed  # Larger = faster interpolation

    def update(self, value: float):
        self._target = float(np.clip(value, 0.0, 1.0))

    def render(self, surface: pygame.Surface, center: tuple):
        # --- Smooth interpolation based on time ---
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        delta = self._target - self._display
        self._display += delta * min(1.0, dt * self._smoothing_speed)

        # --- Clear the region ---
        clear_rect = pygame.Rect(center[0] - self.radius - 10,
                                 center[1] - self.radius - 20,
                                 self.radius * 2 + 20,
                                 self.radius * 2 + 40)
        pygame.draw.rect(surface, (255, 255, 255), clear_rect)

        # --- Draw circle ---
        pygame.draw.circle(surface, (50, 50, 50), center, self.radius, 2)

        # --- Draw ticks ---
        for i in range(0, 11):
            angle = np.pi * (1 - i / 10)
            x1 = int(center[0] + self.radius * 0.85 * np.cos(angle))
            y1 = int(center[1] - self.radius * 0.85 * np.sin(angle))
            x2 = int(center[0] + self.radius * 0.95 * np.cos(angle))
            y2 = int(center[1] - self.radius * 0.95 * np.sin(angle))
            pygame.draw.line(surface, (160, 160, 160), (x1, y1), (x2, y2), 1)

        # --- Needle ---
        val = np.clip(self._display, 0.0, 1.0)
        angle = np.pi * (1 - val)
        needle_x = int(center[0] + self.radius * 0.8 * np.cos(angle))
        needle_y = int(center[1] - self.radius * 0.8 * np.sin(angle))
        pygame.draw.line(surface, (0, 0, 255), center, (needle_x, needle_y), 3)

        # --- Labels ---
        font = pygame.font.SysFont(None, 24)
        low = font.render("Low", True, (0, 0, 0))
        high = font.render("High", True, (0, 0, 0))
        surface.blit(low, (center[0] - self.radius - 35, center[1] - 10))
        surface.blit(high, (center[0] + self.radius + 5, center[1] - 10))

        value_text = font.render(f"{val:.2f}", True, (0, 0, 0))
        surface.blit(value_text, (center[0] - value_text.get_width() // 2, center[1] + self.radius + 5))
