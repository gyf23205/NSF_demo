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

FONT = 'Helvetica'
FONT_SIZE = 20
line_width = 1.5

base_altitude = 200
range_altitude = 150
min_altitude = 0
max_altitude = 2


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
        posy = self.pos[1] + len(self.texts) * self.size * line_width
        self.texts.append((text, (posx, posy)))

    def clear(self):
        self.texts = []


class Background:
    def __init__(self, file_name, bound_x_min, bound_x_max, bound_y_min, bound_y_max):
        self.figure = pygame.image.load(file_name)
        self.surface = self.figure
        self.surface = pygame.transform.scale(self.surface, (bound_x_max, bound_y_max))
        self.rect = self.surface.get_rect()
        self.min_bound = np.array([bound_x_min, bound_y_min])
        self.max_bound = np.array([bound_x_max, bound_y_max])


class GameMgr:
    def __init__(self, mode=1):
        self.initial = True
        self.mode = mode
        self.t0 = 0

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
        self.target_decided = True
        self.new_target = []
        self.new_target_triggered = False

        # Takeoff positions
        self.takeoff_position = []

        # Drone image: Main drones
        drone1 = Drone(file_name=IMAGE_PATH + 'drone1.png', sc=0.1, rt=0.0)
        drone2 = Drone(file_name=IMAGE_PATH + 'drone2.png', sc=0.1, rt=0.0)
        self.objects.append(drone1)
        self.objects.append(drone2)

        # Drone image: side view - altitude
        drone_side1 = Drone(file_name=IMAGE_PATH + 'drone_side1.png', pos_x=1630, pos_y=base_altitude, sc=0.8, rt=0.0)
        drone_side2 = Drone(file_name=IMAGE_PATH + 'drone_side2.png', pos_x=1780, pos_y=base_altitude, sc=0.8, rt=0.0)
        self.objects.append(drone_side1)
        self.objects.append(drone_side2)

        # Background and text interface init
        self.background = Background(file_name=IMAGE_PATH + 'terrain_blur.png',
                                     bound_x_min=0, bound_x_max=1500, bound_y_min=0, bound_y_max=900)
        self.status = Font(FONT, FONT_SIZE, (20, 200))
        self.instruction = Font(FONT, FONT_SIZE, (20, 40))
        self.instruction.update("SIMULATION INSTRUCTION")

        # GUI sub-part 1: altitude
        self.display_navigation = Font(FONT, FONT_SIZE, (1510, 22))
        self.display_navigation.update("                               Navigation")
        self.display_navigation.update("Too High")
        self.display_navigation.update("")
        self.display_navigation.update("Safe")
        self.display_navigation.update("")
        self.display_navigation.update("Too Low")

        # GUI sub-part 2: [mission] 1) update mission / 2) confirm victim
        self.display_mission = Font(FONT, FONT_SIZE, (1510, 250))
        self.display_mission.update("                                  Mission")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("     Accept        Reject               Accept        Reject")

        # Event: victim confirmation
        self.victim_id = [0, 0]
        self.victim_images = [None, None]
        self.victim_detected = [False, False]
        self.victim_clicked = [0, 0]

        # Event message

        # Obstacles

        # Position of objects

        # Input device
        self.mouse_pos = [0, 0]

    def set_target(self, target=None, new_target=None):
        if target is not None:
            self.target = target
        if new_target is not None:
            self.new_target = new_target

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
                # elif event.key == pygame.K_UP:
                #     self.desired_alt_meter = min(1.5, self.desired_alt_meter + 0.05)
                # elif event.key == pygame.K_DOWN:
                #     self.desired_alt_meter = max(0.1, self.desired_alt_meter - 0.05)
                # elif event.key == pygame.K_LEFT or pygame.K_RIGHT:
                #     self.desired_alt_meter = 1
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_pos = pygame.mouse.get_pos()
                # print(self.mouse_pos)

    def correct_victim(self):
        for ind in range(2):
            if self.victim_id[ind] < 7:
                if self.victim_clicked[ind] == 1:
                    print(f'Correct: accepted by drone {int(ind + 1)}')
                elif self.victim_clicked[ind] == 2:
                    print(f'Wrong: rejected by drone {int(ind + 1)}')
            elif self.victim_id[ind] >= 7:
                if self.victim_clicked[ind] == 1:
                    print(f'Wrong: accepted by drone {int(ind + 1)}')
                elif self.victim_clicked[ind] == 2:
                    print(f'Correct: rejected by drone {int(ind + 1)}')

    def input_victim(self):
        if 1520 <= self.mouse_pos[0] <= 1600 and 460 <= self.mouse_pos[1] <= 500:
            self.victim_clicked[0] = 1  # Accepted by drone 1
            self.mouse_pos = [0, 0]
        elif 1610 <= self.mouse_pos[0] <= 1690 and 460 <= self.mouse_pos[1] <= 500:
            self.victim_clicked[0] = 2  # Rejected by drone 1
            self.correct_victim()
            self.mouse_pos = [0, 0]
        elif 1730 <= self.mouse_pos[0] <= 1810 and 460 <= self.mouse_pos[1] <= 500:
            self.victim_clicked[1] = 1  # Accepted by drone 2
            self.correct_victim()
            self.mouse_pos = [0, 0]
        elif 1820 <= self.mouse_pos[0] <= 1900 and 460 <= self.mouse_pos[1] <= 500:
            self.victim_clicked[1] = 2  # Accepted by drone 2
            self.correct_victim()
            self.mouse_pos = [0, 0]
        else:
            self.victim_clicked[0] = 0  # Reset to unselected
            self.victim_clicked[1] = 0

    def input_new_target(self):
        if 1520 <= self.mouse_pos[0] <= 1600 and 460 <= self.mouse_pos[1] <= 500:
            self.target_decided = True
            self.mouse_pos = [0, 0]
        # else:
        #     self.target_decided = False

    def update(self):
        # Initialization
        if self.initial:
            self.t0 = pygame.time.get_ticks()
            self.initial = False

        # Update message handling
        self.status.clear()

        # Update object status
        for i, obj in enumerate(self.objects):
            obj.update()

        # Update text
        time_display = (pygame.time.get_ticks() - self.t0) * 1e-3
        self.status.update('Time: %.1f sec' % time_display)

    def render(self):
        # Background
        self.screen.fill(WHITE)
        self.screen.blit(self.background.surface, self.background.rect)

        # Visual feedback
        # pygame.draw.rect(self.screen, self.visual_code, (10, 658, 100, 100))

        # Objects: drones
        self.objects[0].load(IMAGE_PATH + 'drone1.png')
        self.objects[1].load(IMAGE_PATH + 'drone2.png')

        # Objects: drones - side view
        self.objects[2].load(IMAGE_PATH + 'drone_side1.png')
        self.objects[3].load(IMAGE_PATH + 'drone_side2.png')

        # Altitude information
        pygame.draw.rect(self.screen, (100, 100, 100), (1580, 50, 100, 150))
        pygame.draw.rect(self.screen, (100, 100, 100), (1730, 50, 100, 150))
        pygame.draw.line(self.screen, RED, (1580, self.objects[2].position[1]),
                         (1580 + 100, self.objects[2].position[1]), 3)
        pygame.draw.line(self.screen, BLUE, (1730, self.objects[3].position[1]),
                         (1730 + 100, self.objects[3].position[1]), 3)

        # Victims
        for ind in range(2):
            if self.victim_id[ind] > 0:
                image = 'victim{0}.png'.format(self.victim_id[ind])
                self.victim_images[ind] = Drone(file_name=IMAGE_PATH + image, pos_x=1600 + 210 * ind, pos_y=360,
                                                sc=1.0, rt=0.0)
                self.screen.blit(self.victim_images[ind].surface, self.victim_images[ind].rect)

        # Mouse selections
        pygame.draw.rect(self.screen, (0, 150, 200), (1520, 460, 80, 40))
        pygame.draw.rect(self.screen, (250, 50, 50), (1520 + 90, 460, 80, 40))
        pygame.draw.rect(self.screen, (0, 150, 200), (1730, 460, 80, 40))
        pygame.draw.rect(self.screen, (250, 50, 50), (1730 + 90, 460, 80, 40))

        # Mouse action
        self.input_victim()
        self.input_new_target()

        # Targets
        for i, pos in enumerate(self.target):
            pygame.draw.circle(self.screen, BLUE, pos, 10)
        for i, pos in enumerate(self.new_target):
            pygame.draw.circle(self.screen, RED, pos, 15)

        # Takeoff positions
        for i, pos in enumerate(self.takeoff_position):
            pygame.draw.circle(self.screen, BLACK, pos, 10)

        # Objects (drones)
        for i, obj in enumerate(self.objects):
            self.screen.blit(obj.surface, obj.rect)

        # font interface
        for text in self.instruction.texts:
            self.screen.blit(text[0], text[1])
        for text in self.status.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_navigation.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_mission.texts:
            self.screen.blit(text[0], text[1])

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

