import sys
import pygame
import numpy as np

# Patch sys.path to import from examples directory
import os
sys.path.append(os.path.dirname(__file__))

from gui_panel import GameMgr

def main():
    pygame.init()
    mgr = GameMgr()
    # Simulate some drone positions for awareness update
    drone_positions = [[100, 100], [200, 200]]
    mgr.set_takeoff_positions([[50, 50], [850, 650]])
    mgr.set_target(target=[[300, 300], [400, 400]], new_target=[[500, 500]])
    mgr.set_wind([1.0, 1.0, 20], meter=True)
    running = True
    frame = 0
    while running and frame < 10:
        mgr.input()
        mgr.update_awareness(drone_positions)
        mgr.update()
        mgr.render()
        pygame.time.delay(100)
        frame += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    pygame.quit()

if __name__ == "__main__":
    main()
