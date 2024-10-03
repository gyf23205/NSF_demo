import pygame
import os
from drone import *


BOUND_X_MAX = 1920
BOUND_Y_MAX = 900
IMAGE_PATH = 'images/'

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
SEE = (0, 119, 190)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)


class GameMgr:
    def __init__(self, mode=1):
        self.initial = True
        self.mode = mode

        # Game screen initialization
        pygame.font.init()

        # Window position
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)
        self.screen = pygame.display.set_mode((BOUND_X_MAX, BOUND_Y_MAX), 0)

        # Game title on the window
        pygame.display.set_caption('Drone SAR Mission')

        # game clock initialization
        self.clock = pygame.time.Clock()
        self.obstacle_counter = 0

        # object queue initialization
        self.objects = []
        self.collides = []

        # Targets
        self.target = []

        # Takeoff positions
        self.takeoff_position = []

        # Drone image
        drone1 = Drone(file_name=IMAGE_PATH + 'drone1.png', sc=0.1, rt=0.0)
        drone2 = Drone(file_name=IMAGE_PATH + 'drone2.png', sc=0.1, rt=0.0)
        self.objects.append(drone1)
        self.objects.append(drone2)

        # Background and text interface init
        # self.background = Background(file_name=IMAGE_PATH + 'Purdue_blurred.png', moving_name=IMAGE_PATH + 'road.png',
        #                              bound_x_min=BOUND_X_MIN, bound_x_max=BOUND_X_MAX, bound_y_min=BOUND_Y_MIN,
        #                              bound_y_max=BOUND_Y_MAX)
        # self.status = Font(FONT, FONT_SIZE, (20, 200))
        # self.instruction = Font(FONT, FONT_SIZE, (20, 40))
        # self.instruction.update("SIMULATION INSTRUCTION")
        # self.instruction.update("Input device: %s" % control)

        # Event message

        # Obstacles

        # Position of objects

        # Input device
        self.desired_pos_meter = [0, 0]
        self.desired_alt_meter = 1

    def set_target(self, target):
        self.target = target

    def set_takeoff_positions(self, position):
        self.takeoff_position = position

    def input(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mode = 0
                    print('Escape Command')
                elif event.key == pygame.K_SPACE:
                    self.mode = 2
                    print('Landing Command')
                elif event.key == pygame.K_UP:
                    self.desired_alt_meter = min(1.5, self.desired_alt_meter + 0.05)
                elif event.key == pygame.K_DOWN:
                    self.desired_alt_meter = max(0.1, self.desired_alt_meter - 0.05)
                elif event.key == pygame.K_LEFT or pygame.K_RIGHT:
                    self.desired_alt_meter = 1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                desired_pos_x = -(mouse_pos[0] - 0.5 * BOUND_X_MAX) / 300.0
                desired_pos_y = (mouse_pos[1] - 0.5 * BOUND_Y_MAX) / 300.0
                desired_pos_meter = [desired_pos_x, desired_pos_y]
                print(desired_pos_meter)
                self.desired_pos_meter = desired_pos_meter

    def update(self):
        # Update object status
        for i, obj in enumerate(self.objects):
            obj.update()

    def render(self):
        # Background
        self.screen.fill(WHITE)
        # self.screen.blit(self.background.surface, self.background.rect)

        # Visual feedback
        # pygame.draw.rect(self.screen, self.visual_code, (10, 658, 100, 100))

        # Rotate
        # self.objects[0].rt = self.objects[0].attitude * 180.0 / np.pi
        self.objects[0].load(IMAGE_PATH + 'drone1.png')
        self.objects[1].load(IMAGE_PATH + 'drone2.png')

        # Targets
        for i, pos in enumerate(self.target):
            pygame.draw.circle(self.screen, BLUE, pos, 10)

        # Takeoff positions
        for i, pos in enumerate(self.takeoff_position):
            pygame.draw.circle(self.screen, BLACK, pos, 10)

        # Objects (drones)
        for i, obj in enumerate(self.objects):
            self.screen.blit(obj.surface, obj.rect)

        # font interface
        # for text in self.instruction.texts:
        #     self.screen.blit(text[0], text[1])
        # for text in self.status.texts:
        #     self.screen.blit(text[0], text[1])

        # record sign
        # if self.record and self.record_sign > 0:
        #     pygame.draw.circle(self.screen, RED, (20, 20), 10)

        # double buffer update
        pygame.display.flip()

        # ??
        pygame.display.update()

        # Pausing
        # if self.initial:
        #     # Wait time (mil-sec)
        #     pygame.time.delay(2500)
        #     self.initial = False
        #     self.t0 = pygame.time.get_ticks()

        # Text info
        # if not self.mode and not self.collision:
        #     self.trial += 1
        #     display_surface = pygame.display.set_mode((BOUND_X_MAX, BOUND_Y_MAX))
        #     font = pygame.font.Font('freesansbold.ttf', 32)
        #     text = font.render('Trial %d: Safe landing at %.2f sec' % (self.trial, self.final_time_record), True, WHITE, BLACK)
        #     textrect = text.get_rect()
        #     textrect.center = (BOUND_X_MAX // 2, BOUND_Y_MAX // 2)
        #     display_surface.fill(BLACK)
        #     display_surface.blit(text, textrect)
        #     pygame.display.update()
        #     pygame.time.delay(2500)
        # elif not self.mode and self.collision:
        #     self.trial += 1
        #     display_surface = pygame.display.set_mode((BOUND_X_MAX, BOUND_Y_MAX))
        #     font = pygame.font.Font('freesansbold.ttf', 32)
        #     text = font.render('Trial %d: Crash at %.2f sec' % (self.trial, self.final_time_record), True,
        #                        RED, BLACK)
        #     textrect = text.get_rect()
        #     textrect.center = (BOUND_X_MAX // 2, BOUND_Y_MAX // 2)
        #     display_surface.fill(BLACK)
        #     display_surface.blit(text, textrect)
        #     pygame.display.update()
        #     pygame.time.delay(2500)

