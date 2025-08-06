import pygame
import numpy as np
import time

class RealtimeHeartPlot:
    def __init__(self, width=400, height=100, sweep_duration=7.0, position=(0, 0), hr_interval_seconds=5):
        self.width = width
        self.height = height
        self.position = position
        self.sweep_duration = sweep_duration
        self.hr_interval_seconds = hr_interval_seconds

        self._data = None
        self._start_time = None
        self._last_hr_time = time.time()
        self._sweeping = False
        self._points = []

        self.surface = pygame.Surface((self.width, self.height))
        self.surface.fill((255, 255, 255))

    def update(self, new_data: np.ndarray):
        if self._sweeping:
            return  # Ignore new input during active sweep

        self._data = new_data
        self._start_time = time.time()
        self._sweeping = True
        self._points = []
        self.surface.fill((255, 255, 255))  # Clear canvas

    def render(self, target_surface: pygame.Surface):
        if not self._sweeping or self._data is None:
            target_surface.blit(self.surface, self.position)
            return

        now = time.time()
        elapsed = now - self._start_time
        portion = min(1.0, elapsed / self.sweep_duration)
        num_to_draw = int(len(self._data) * portion)

        self.surface.fill((255, 255, 255))  # clear canvas

        # Label
        font = pygame.font.SysFont(None, 24)
        label = font.render("Heart rate (ECG)", True, (0, 0, 0))
        self.surface.blit(label, (10, 5))

        # Leave padding for label
        top_padding = label.get_height() + 10
        plot_height = self.height - top_padding

        # Y scale based on actual data
        y_min = float(np.min(self._data))
        y_max = float(np.max(self._data))
        y_range = y_max - y_min if y_max > y_min else 1e-6

        points = []
        for i in range(num_to_draw):
            x = int(i * self.width / len(self._data))
            val = self._data[i]
            y = int(plot_height - ((val - y_min) / y_range) * plot_height)
            y += top_padding  # shift down so it doesn't touch label
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(self.surface, (0, 255, 0), False, points, 2)

        if portion >= 1.0:
            self._sweeping = False

        target_surface.blit(self.surface, self.position)


