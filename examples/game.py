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


class Mission:
    def __init__(self):
        self.accepted = 0
        self.rejected = 0
        self.response_time = []
        self.correctness = []


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
        self.target_clicked = 0
        self.target_decided = True
        self.new_target = []
        self.new_target_triggered = False

        # Winds
        self.wind = []

        # Takeoff positions
        self.takeoff_position = []

        # Numbers for mission: finding survivors
        self.missions = [Mission(), Mission()]

        # Health
        self.health = [100, 100]

        # Survey
        self.workload = 0

        # Drone image: Main drones
        drone1 = Drone(file_name=IMAGE_PATH + 'drone1.png', sc=0.1, rt=0.0)
        drone2 = Drone(file_name=IMAGE_PATH + 'drone2.png', sc=0.1, rt=0.0)
        self.objects.append(drone1)
        self.objects.append(drone2)

        # Drone image: side view - altitude
        drone_side1 = Drone(file_name=IMAGE_PATH + 'drone_side1.png', pos_x=1350, pos_y=base_altitude, sc=0.8, rt=0.0)
        drone_side2 = Drone(file_name=IMAGE_PATH + 'drone_side2.png', pos_x=1490, pos_y=base_altitude, sc=0.8, rt=0.0)
        self.objects.append(drone_side1)
        self.objects.append(drone_side2)

        # Drone image: side view - Mission re-planning
        drone_small1 = Drone(file_name=IMAGE_PATH + 'drone1.png', pos_x=1400, pos_y=250 + 80, sc=0.05, rt=0.0)
        drone_small2 = Drone(file_name=IMAGE_PATH + 'drone2.png', pos_x=1400, pos_y=250 + 80 + 50, sc=0.05, rt=0.0)
        self.objects.append(drone_small1)
        self.objects.append(drone_small2)

        # Background and text interface init
        self.background = Background(file_name=IMAGE_PATH + 'terrain_blur.png',
                                     bound_x_min=0, bound_x_max=1200, bound_y_min=0, bound_y_max=720)
        self.status = Font(FONT, FONT_SIZE, (20, 200))
        self.instruction = Font(FONT, FONT_SIZE, (20, 40))
        self.instruction.update("SIMULATION INSTRUCTION")

        # GUI sub-part 1: altitude
        self.display_navigation = Font(FONT, FONT_SIZE, (1220, 22))
        self.display_navigation.update("                                Navigation")
        self.display_navigation.update("Too High")
        self.display_navigation.update("")
        self.display_navigation.update("Safe")
        self.display_navigation.update("")
        self.display_navigation.update("Too Low")

        # GUI sub-part 2: [mission] 1) update mission / 2) confirm victim
        self.display_mission = Font(FONT, FONT_SIZE, (1320, 470))
        self.display_mission.update("                                    Confirm Victims")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("")
        self.display_mission.update("Accept        Reject                                        Accept        Reject")

        # GUI sub-part 3: [guidance] re-planning
        x_rep = [1370, 250]
        self.display_planning = Font(FONT, FONT_SIZE, (x_rep[0], x_rep[1]))
        self.display_planning.update("                        Mission Re-planning")
        self.display_planning.update("Decision   Total Dist.   Drone 1 Dist.   Drone 2 Dist.")
        # Column-wise
        self.display_planning_c1 = Font(FONT, FONT_SIZE, (x_rep[0] + 100, x_rep[1] + 65))
        self.display_planning_c2 = Font(FONT, FONT_SIZE, (x_rep[0] + 100 + 1 * 100, x_rep[1] + 65))
        self.display_planning_c3 = Font(FONT, FONT_SIZE, (x_rep[0] + 100 + 2 * 100, x_rep[1] + 65))
        # Values
        self.planning_distances = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # GUI sub-part 4: [fault (wind)]
        self.display_fault = Font(FONT, FONT_SIZE, (30, 740))
        self.display_fault.update("                  Wind Response")
        self.display_fault.update("")
        self.display_fault.update("Change routes             Maintain routes")
        self.display_fault.update("Maintain speed            Slow-down speed")

        # GUI sub-part 5: [Mission monitor]
        self.display_m_status = Font(FONT, FONT_SIZE, (370, 740))

        # GUI sub-part 6: [Mission report]
        self.display_m_report = Font(FONT, FONT_SIZE, (970, 740))
        self.display_m_report.update("    Mission Report")
        value = np.random.randint(low=3, high=6)
        self.display_m_report.update("Found >= %d survivors?" % value)
        self.display_m_report.update("")
        self.display_m_report.update("YES                 NO")

        # GUI sub-part 7: [Drone health]
        self.display_health = Font(FONT, FONT_SIZE, (1680, 22))
        self.display_health.update("Drone Health")

        # Event: victim confirmation
        self.victim_id = [0, 0]
        self.victim_images = [None, None]
        self.victim_detected = [False, False]
        self.victim_clicked = [0, 0]
        self.victim_block_choice = [False, False]
        self.victim_timing = [0, 0]

        # Event: wind
        self.wind_danger = False
        self.wind_triggered = False
        self.wind_decided = True
        self.wind_clicked = 0

        # Mission report
        self.report_triggered = False
        self.report_requested = False
        self.report_clicked = 0

        # Input device
        self.mouse_pos = [0, 0]

    def set_target(self, target=None, new_target=None):
        if target is not None:
            self.target = target
        if new_target is not None:
            self.new_target = new_target

    def set_wind(self, wind):
        self.wind.append(wind)

    def reset_wind(self):
        self.wind = []

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
                print(self.mouse_pos)

    def correct_victim(self):
        for ind in range(2):
            if 0 < self.victim_id[ind] < 15:
                if self.victim_clicked[ind] == 1:
                    self.missions[ind].accepted += 1
                    self.missions[ind].correctness.append(1)
                    print(f'Correct: accepted by drone {int(ind + 1)}')
                    print(f'Correctness by drone {int(ind + 1)}: {self.missions[ind].correctness}')
                elif self.victim_clicked[ind] == 2:
                    self.missions[ind].rejected += 1
                    self.missions[ind].correctness.append(0)
                    print(f'Wrong: rejected by drone {int(ind + 1)}')
                    print(f'Correctness by drone {int(ind + 1)}: {self.missions[ind].correctness}')
            elif self.victim_id[ind] >= 15:
                if self.victim_clicked[ind] == 1:
                    self.missions[ind].accepted += 1
                    self.missions[ind].correctness.append(0)
                    print(f'Wrong: accepted by drone {int(ind + 1)}')
                    print(f'Correctness by drone {int(ind + 1)}: {self.missions[ind].correctness}')
                elif self.victim_clicked[ind] == 2:
                    self.missions[ind].rejected += 1
                    self.missions[ind].correctness.append(1)
                    print(f'Correct: rejected by drone {int(ind + 1)}')
                    print(f'Correctness by drone {int(ind + 1)}: {self.missions[ind].correctness}')

    def mouse_actions(self):
        # Victim
        if 1310 <= self.mouse_pos[0] <= 1390 and 825 <= self.mouse_pos[1] <= 865:
            self.victim_clicked[0] = 1  # Accepted by drone 1
            self.correct_victim()
        elif 1400 <= self.mouse_pos[0] <= 1480 and 825 <= self.mouse_pos[1] <= 865:
            self.victim_clicked[0] = 2  # Rejected by drone 1
            self.correct_victim()
        elif 1640 <= self.mouse_pos[0] <= 1720 and 825 <= self.mouse_pos[1] <= 865:
            self.victim_clicked[1] = 1  # Accepted by drone 2
            self.correct_victim()
        elif 1730 <= self.mouse_pos[0] <= 1810 and 825 <= self.mouse_pos[1] <= 865:
            self.victim_clicked[1] = 2  # Accepted by drone 2
            self.correct_victim()
        # New target
        elif 1370 <= self.mouse_pos[0] <= 1745 and 310 <= self.mouse_pos[1] <= 355:
            self.target_clicked = 1
        elif 1370 <= self.mouse_pos[0] <= 1745 and 360 <= self.mouse_pos[1] <= 405:
            self.target_clicked = 2
        # Wind
        elif 45 <= self.mouse_pos[0] <= 135 and 790 <= self.mouse_pos[1] <= 870:
            self.wind_clicked = 1
        elif 145 <= self.mouse_pos[0] <= 235 and 790 <= self.mouse_pos[1] <= 870:
            self.wind_clicked = 2
        # Mission report
        elif 935 <= self.mouse_pos[0] <= 1035 and 800 <= self.mouse_pos[1] <= 880:
            self.report_clicked = 1
        elif 1050 <= self.mouse_pos[0] <= 1150 and 800 <= self.mouse_pos[1] <= 880:
            self.report_clicked = 2
        # Reset clicked
        else:
            self.victim_clicked[0] = 0  # Reset to unselected
            self.victim_clicked[1] = 0
            self.target_clicked = 0
            self.wind_clicked = 0

        # Reset mouse position
        self.mouse_pos = [0, 0]

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

        # Re-planning text
        self.display_planning_c1.clear()
        self.display_planning_c1.update('%.1f' % self.planning_distances[0])
        self.display_planning_c1.update("")
        self.display_planning_c1.update('%.1f' % self.planning_distances[3])
        self.display_planning_c2.clear()
        self.display_planning_c2.update('%.1f' % self.planning_distances[1])
        self.display_planning_c2.update("")
        self.display_planning_c2.update('%.1f' % self.planning_distances[4])
        self.display_planning_c3.clear()
        self.display_planning_c3.update('%.1f' % self.planning_distances[2])
        self.display_planning_c3.update("")
        self.display_planning_c3.update('%.1f' % self.planning_distances[5])

        # Mission status update
        self.display_m_status.clear()
        a0 = self.missions[0].accepted
        a1 = self.missions[1].accepted
        r0 = self.missions[0].rejected
        r1 = self.missions[1].rejected
        t0 = np.mean(self.missions[0].response_time) if self.missions[0].response_time else 0.0
        t1 = np.mean(self.missions[1].response_time) if self.missions[1].response_time else 0.0
        self.display_m_status.update("                                  Mission Status: Survivors")
        self.display_m_status.update("                   Accepted     |     Rejected     |    Avg. Response Time")
        self.display_m_status.update(
            "Drone 1:             %d                      %d                       %.2f sec" % (a0, r0, t0))
        self.display_m_status.update(
            "Drone 2:             %d                      %d                       %.2f sec" % (a1, r1, t1))

    def render(self):
        # Background
        self.screen.fill(WHITE)
        self.screen.blit(self.background.surface, self.background.rect)

        # Objects: drones
        self.objects[0].load(IMAGE_PATH + 'drone1.png')
        self.objects[1].load(IMAGE_PATH + 'drone2.png')

        # Objects: drones - side view
        self.objects[2].load(IMAGE_PATH + 'drone_side1.png')
        self.objects[3].load(IMAGE_PATH + 'drone_side2.png')

        # Objects: drones - small ones
        x_rep = [1370, 310]
        if self.target_decided:
            pygame.draw.rect(self.screen, BLACK, (x_rep[0], x_rep[1], 375, 110))
        else:
            pygame.draw.rect(self.screen, (0, 255, 0), (x_rep[0], x_rep[1], 375, 45))
            pygame.draw.rect(self.screen, YELLOW, (x_rep[0], x_rep[1] + 50, 375, 45))
            # Objects (drones - small ones)
            self.objects[4].load(IMAGE_PATH + 'drone1.png')
            self.objects[5].load(IMAGE_PATH + 'drone2.png')
            for i, obj in enumerate(self.objects[4:6]):
                self.screen.blit(obj.surface, obj.rect)

        # Altitude information
        x_alt = [1300, 1440]
        pygame.draw.rect(self.screen, (130, 130, 130), (x_alt[0], 50, 100, 150))
        pygame.draw.rect(self.screen, (130, 130, 130), (x_alt[1], 50, 100, 150))
        pygame.draw.line(self.screen, RED, (x_alt[0], self.objects[2].position[1]),
                         (x_alt[0] + 100, self.objects[2].position[1]), 3)
        pygame.draw.line(self.screen, BLUE, (x_alt[1], self.objects[3].position[1]),
                         (x_alt[1] + 100, self.objects[3].position[1]), 3)

        # Drone health
        x_hea = [1610, 1750]
        pygame.draw.rect(self.screen, BLACK, (x_hea[0], 50, 100, 150))
        pygame.draw.rect(self.screen, BLACK, (x_hea[1], 50, 100, 150))
        # Random process
        random1 = np.random.uniform()
        random2 = np.random.uniform()
        reduce1 = 0.03 if random1 > 0.01 else 0.0
        reduce2 = 0.03 if random2 > 0.01 else 0.0
        self.health[0] -= reduce1
        self.health[1] -= reduce2
        # Position
        h_start = [-1.5 * self.health[0] + 200, -1.5 * self.health[1] + 200]
        h_len = [max(0, 200 - int(h_start[0])), max(0, 200 - int(h_start[1]))]
        pygame.draw.rect(self.screen, (20, 230, 70), (x_hea[0], int(h_start[0]), 100, h_len[0]))
        pygame.draw.rect(self.screen, (20, 230, 70), (x_hea[1], int(h_start[1]), 100, h_len[1]))

        # Victims
        for ind in range(2):
            if self.victim_id[ind] > 0:
                image = 'victim{0}.jpeg'.format(self.victim_id[ind])
                self.victim_images[ind] = Drone(file_name=IMAGE_PATH + image, pos_x=1385 + 350 * ind, pos_y=660,
                                                sc=0.25, rt=0.0)
                self.screen.blit(self.victim_images[ind].surface, self.victim_images[ind].rect)

        # Mouse selections: victim
        x_vic = [1310, 1640]
        y_vic = 825
        pygame.draw.rect(self.screen, (0, 150, 200), (x_vic[0], y_vic, 80, 40))
        pygame.draw.rect(self.screen, (250, 50, 50), (x_vic[0] + 90, y_vic, 80, 40))
        pygame.draw.rect(self.screen, (0, 150, 200), (x_vic[1], y_vic, 80, 40))
        pygame.draw.rect(self.screen, (250, 50, 50), (x_vic[1] + 90, y_vic, 80, 40))
        if self.victim_block_choice[0]:
            pygame.draw.rect(self.screen, (0, 0, 0), (x_vic[0], y_vic, 170, 40))
        if self.victim_block_choice[1]:
            pygame.draw.rect(self.screen, (0, 0, 0), (x_vic[1], y_vic, 170, 40))

        # Mouse selections: wind
        if not self.wind_decided:
            pygame.draw.rect(self.screen, (0, 255, 0), (20, 790, 140, 80))
            pygame.draw.rect(self.screen, YELLOW, (195, 790, 140, 80))
        else:
            pygame.draw.rect(self.screen, BLACK, (15, 790, 320, 80))

        # Wind
        for i, value in enumerate(self.wind):
            pygame.draw.circle(self.screen, (35, 250, 252), [value[0], value[1]], value[2])

        # Mission report
        if self.report_requested:
            pygame.draw.rect(self.screen, (0, 150, 200), (935, 800, 100, 80))
            pygame.draw.rect(self.screen, (250, 50, 50), (1050, 800, 100, 80))
        else:
            pygame.draw.rect(self.screen, BLACK, (935, 770, 215, 110))

        # Mouse actions
        self.mouse_actions()

        # Targets
        for i, pos in enumerate(self.target):
            pygame.draw.circle(self.screen, BLUE, pos, 10)
        for i, pos in enumerate(self.new_target):
            pygame.draw.circle(self.screen, RED, pos, 15)

        # Takeoff positions
        for i, pos in enumerate(self.takeoff_position):
            pygame.draw.circle(self.screen, BLACK, pos, 10)

        # Objects (drones)
        for i, obj in enumerate(self.objects[0:4]):
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
        for text in self.display_planning.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_planning_c1.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_planning_c2.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_planning_c3.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_fault.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_m_status.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_m_report.texts:
            self.screen.blit(text[0], text[1])
        for text in self.display_health.texts:
            self.screen.blit(text[0], text[1])

        # record sign
        # if self.record and self.record_sign > 0:
        #     pygame.draw.circle(self.screen, RED, (20, 20), 10)

        # double buffer update
        pygame.display.flip()

        # ??
        pygame.display.update()

    def mouse_survey(self):
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.mouse_pos = pygame.mouse.get_pos()
                print(self.mouse_pos)

        if 560 <= self.mouse_pos[0] <= 760 and 400 <= self.mouse_pos[1] <= 600:
            self.workload = 1
            print('Reported Workload: LOW')
        elif 860 <= self.mouse_pos[0] <= 1060 and 400 <= self.mouse_pos[1] <= 600:
            self.workload = 2
            print('Reported Workload: MEDIUM')
        elif 1160 <= self.mouse_pos[0] <= 1360 and 400 <= self.mouse_pos[1] <= 600:
            self.workload = 3
            print('Reported Workload: HIGH')

    def survey_render(self):
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Survey: Workload', True, WHITE, BLACK)
        text_rect = text.get_rect()
        text_rect.center = (BOUND_X_MAX // 2, BOUND_Y_MAX // 3)
        self.screen.fill(BLACK)
        self.screen.blit(text, text_rect)

        # Survey buttons
        pygame.draw.rect(self.screen, WHITE, (560, 400, 200, 200))
        pygame.draw.rect(self.screen, WHITE, (860, 400, 200, 200))
        pygame.draw.rect(self.screen, WHITE, (1160, 400, 200, 200))
        button = font.render('LOW                      MEDIUM                      HIGH', True, RED)
        button_rect = text.get_rect()
        button_rect.center = (760, 500)
        self.screen.blit(button, button_rect)

        # Check mouse action
        self.mouse_survey()

        # Update
        pygame.display.flip()
        pygame.display.update()

        # Escape
        if self.workload > 0:
            self.mode = 4

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
