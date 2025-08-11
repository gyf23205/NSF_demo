import pygame
import os
from examples.vehicles import *
from constants import *
from util_classes import Font, Button, Bar
from pygame.font import SysFont
from shapely.geometry import LineString, box


class DroneHealth:
    def __init__(self, screen, pos, virtual_drone, workspace):
        self.screen = screen
        self.x0, self.y0 = pos
        self.drone = virtual_drone
        self.grid_width = 100
        self.grid_height = line_height * FONT_SIZE
        self.spacing = 20
        self.idx_txt = SysFont(FONT, FONT_SIZE)
        self.alt_bar = Bar(screen, (self.x0 + self.grid_width + self.spacing, self.y0, self.grid_width, self.grid_height))
        self.health_bar = Bar(screen, (self.x0 + 2*(self.grid_width + self.spacing), self.y0, self.grid_width, self.grid_height))
        self.current_pos_txt = SysFont(FONT, FONT_SIZE)
        self.current_target = SysFont(FONT, FONT_SIZE)
        self.status_txt = SysFont(FONT, FONT_SIZE)
        self.latency_txt = SysFont(FONT, FONT_SIZE)
        self.ws = workspace

    def draw(self):
        # Drone ID
        idx_txt = self.idx_txt.render('         ' + str(self.drone.idx), True, BLACK)
        self.screen.blit(idx_txt, (self.x0, self.y0, self.grid_width, self.grid_height))

        # Position (grid coords)
        gx, gy = self.ws.game_mgr_to_grid(self.drone.position[:2])
        pos_str = f"({gx:.0f}, {gy:.0f})"
        pos_txt = self.current_pos_txt.render(pos_str, True, BLACK)
        self.screen.blit(pos_txt, (self.x0 + 1*(self.grid_width+self.spacing), self.y0, self.grid_width, self.grid_height))

        # Task
        task_str = getattr(self.drone, "status", "Idle")
        task_txt = self.status_txt.render(task_str, True, BLACK)
        self.screen.blit(task_txt, (self.x0 + 3*(self.grid_width+self.spacing), self.y0, self.grid_width, self.grid_height))

        # === Latency bar (left→right), colored by thresholds ===
        lat_ms = float(getattr(self.drone, 'latency_ms', 110.0))

        # Normalize to 0..1 using same bounds as main
        L_MIN, L_MAX = 40.0, 500.0
        frac = (lat_ms - L_MIN) / (L_MAX - L_MIN)
        frac = max(0.0, min(1.0, frac))

        # Color thresholds
        if lat_ms <= 120.0:
            col = (34, 139, 34)    # green
        elif lat_ms <= 250.0:
            col = (255, 165, 0)    # yellow
        else:
            col = (220, 20, 60)    # red

        # Column placement (Latency is the 3rd column)
        bx = self.x0 + 2 * (self.grid_width + self.spacing)
        by = self.y0
        bw = self.grid_width
        bh = self.grid_height

        # Outline
        pygame.draw.rect(self.screen, BLACK, pygame.Rect(bx, by, bw, bh), width=1)

        # Fill amount (leave a 1px border)
        filled_w = int(bw * frac)
        if filled_w > 0:
            pygame.draw.rect(self.screen, col, pygame.Rect(bx + 1, by + 1, max(0, filled_w - 2), bh - 2))

        # Optional tiny text (e.g., "137 ms")
        ms_surf = self.status_txt.render(f"{int(lat_ms)} ms", True, BLACK)
        self.screen.blit(ms_surf, (bx + 6, by))


class GVHealth:
    def __init__(self, screen, pos, virtual_gv, workspace):
        self.screen = screen
        self.x0, self.y0 = pos
        self.gv = virtual_gv
        self.grid_width = 100
        self.grid_height = line_height * FONT_SIZE
        self.spacing = 20
        self.idx_txt = SysFont(FONT, FONT_SIZE)
        self.carrying_txt = SysFont(FONT, FONT_SIZE)
        self.health_bar = Bar(screen, (self.x0 + (self.grid_width + self.spacing), self.y0, self.grid_width, self.grid_height))
        self.current_pos_txt = SysFont(FONT, FONT_SIZE)
        self.current_target = SysFont(FONT, FONT_SIZE)
        self.ws = workspace
        self.status_txt = SysFont(FONT, FONT_SIZE)

    def draw(self):
        # GV ID
        idx_txt = self.idx_txt.render('         ' + str(self.gv.idx), True, BLACK)
        self.screen.blit(idx_txt, (self.x0, self.y0, self.grid_width, self.grid_height))

        # Carrying
        carrying_txt = self.carrying_txt.render(str(getattr(self.gv, 'carrying', False)), True, BLACK)
        self.screen.blit(carrying_txt, (self.x0 + 1*(self.grid_width+self.spacing), self.y0, self.grid_width, self.grid_height))

        # Position
        gx, gy = self.ws.game_mgr_to_grid(self.gv.position[:2])
        pos_str = f"({gx:.0f}, {gy:.0f})"
        pos_txt = self.current_pos_txt.render(pos_str, True, BLACK)
        self.screen.blit(pos_txt, (self.x0 + 2*(self.grid_width+self.spacing), self.y0, self.grid_width, self.grid_height))

        # Task
        task_str = getattr(self.gv, "status", "Idle")
        task_txt = self.status_txt.render(task_str, True, BLACK)
        self.screen.blit(task_txt, (self.x0 + 3*(self.grid_width+self.spacing), self.y0, self.grid_width, self.grid_height))

class SpecialRegions:
    def __init__(self, screen):
        self.screen = screen
        self.regions = [] # Each region is a tuple (task_id, center, radius, color)
        self.num2color_dict = {
            1: ORANGE,  # Orange
            2: RED,   # Red
            3: SEE
        }
        self.color2num_dict = {v: k for k, v in self.num2color_dict.items()}

    def draw(self):
        for i in range(len(self.regions)):
            id, center, radius, color = self.regions[i]
            surface = pygame.Surface((2*radius, 2*radius), pygame.SRCALPHA)
            pygame.draw.circle(surface, color + (100,), (radius, radius), radius)
            self.screen.blit(surface, (center[0] - radius, center[1] - radius))
        
    def add_region(self, id, center, radius, priority):
        # Ensure the new region does not overlap with existing ones
        color = self.num2color_dict[priority]
        # for existing_id, existing_center, existing_radius, _ in self.regions:
        #     if np.linalg.norm(np.array(center) - np.array(existing_center)) < (radius + existing_radius):
        #         print("Region overlaps with an existing region. Not added.")
        #         return

        # If no overlap, add the new region
        self.regions.append((id, np.array(center), radius, color))

    def remove_region(self, idx):
        for i, r in enumerate(self.regions):
            if r[0] == idx:
                self.regions.pop(i)

    def check_tasks_priority(self, tasks):
        priority = [0 for _ in range(len(tasks))]  # Default priority is 0 (no special region)
        # Check if any task is within any special region
        for task in tasks:
            for center, radius, color in self.regions:
                if np.linalg.norm(task[1] - center) < radius:
                    priority[task[0] - 1] = self.color2num_dict[color]
        return priority

class Background:
    def __init__(self, file_name, bound_x_min, bound_x_max, bound_y_min, bound_y_max):
        self.figure = pygame.image.load(file_name).convert()
        self.surface = self.figure
        self.surface = pygame.transform.scale(self.surface, (bound_x_max - bound_x_min, bound_y_max - bound_y_min))
        self.rect = self.surface.get_rect()
        self.rect.topleft = (bound_x_min, bound_y_min)
        # print(self.rect)
        self.min_bound = np.array([bound_x_min, bound_y_min])
        self.max_bound = np.array([bound_x_max, bound_y_max])

class BackgroundNoScale:
    def __init__(self, file_name, bound_x_min, bound_x_max, bound_y_min, bound_y_max):
        self.figure = pygame.image.load(file_name).convert()
        self.surface = self.figure
        self.rect = self.surface.get_rect()
        self.rect.topleft = (bound_x_min, bound_y_min)
        # print(self.rect)
        self.min_bound = np.array([bound_x_min, bound_y_min])
        self.max_bound = np.array([bound_x_max, bound_y_max])


# class EnvironmentInfo:
#     def __init__(self, screen):
#         self.screen = screen
#         self.x0, self.y0 = 1150, 700
#         self.title = Font(FONT, FONT_SIZE, (self.x0, self.y0))
#         self.spacing = '               '
#         self.title.update('                 Environment Info')
#         self.title.update('Location' + self.spacing + self.spacing + '          Speed')
#         self.content = Font(FONT, FONT_SIZE, (self.x0, self.y0 + 2 * int(FONT_SIZE * line_height)))

#     def draw(self, wind):
#         for text, pos in self.title.texts:
#             self.screen.blit(text, pos)
#         self.content.clear()
#         for i, w in enumerate(wind):
#             content = f'Wind {i+1}: ({w[0]:.2f}, {w[1]:.2f}){self.spacing}{w[2]:.2f}'
#             self.content.update(content)
#         for text, pos in self.content.texts:
#             self.screen.blit(text, pos)


class HumanWorkload:
    def __init__(self, screen, pos, virtual_human):
        self.screen = screen
        self.x0, self.y0 = pos
        self.human = virtual_human
        self.grid_width  = 100
        self.grid_height = line_height * FONT_SIZE
        self.spacing     = 20
        self.task_txt = SysFont(FONT, FONT_SIZE)

        # ID label
        self.idx_txt    = SysFont(FONT, FONT_SIZE)
        # utilization bar (0–100%)
        self.util_bar   = Bar(screen,
                              (self.x0 + self.grid_width + self.spacing,
                               self.y0,
                               self.grid_width,
                               self.grid_height))
        # text for percent
        self.pct_txt    = SysFont(FONT, FONT_SIZE)

    def draw(self):
        # 1) draw human label (Agent.label is e.g. "H0", "H1")
        label = getattr(self.human, 'label', '<unknown>')
        idx_txt = self.idx_txt.render(label, True, BLACK)
        self.screen.blit(idx_txt, (self.x0, self.y0))

        # 2) draw utilization bar
        pct = getattr(self.human, 'utilization', 0)
        self.util_bar.draw(pct)

        # 3) draw percent text to the right of the bar
        pct_label = self.pct_txt.render(f'{pct}%', True, BLACK)
        bx = self.x0 + self.grid_width + self.spacing
        self.screen.blit(pct_label, (bx + self.grid_width + 5, self.y0))

        # Task (humanized)
        task_str = getattr(self.human, 'status', 'Idle')
        task_surf = self.task_txt.render(task_str, True, BLACK)
        self.screen.blit(task_surf, (self.x0 + 2*(self.grid_width + self.spacing) + 60, self.y0))


class GameMgr:
    def __init__(self, drones, gvs, humans, ws):
        pygame.init()
        self.t0 = 0
        self.initial = True
        self.drones = drones
        self.gvs = gvs
        self.humans = humans
        self.n_drones = len(drones)
        self.n_gvs = len(gvs)

        # ====== Panel rectangles (black outlines) ======
        self.panel_drones   = pygame.Rect(1135, 10,  775, 290)
        self.panel_gvs      = pygame.Rect(1135, 310, 775, 200)
        self.panel_humans   = pygame.Rect(1135, 520, 775, 180)
        self.panel_legend   = pygame.Rect(1135, 710, 775, 180)
        
        # Define position of blocks, tf=topleft, c=center
        # self.tf_map = (100, 100)
        self.c_drone_icon    = (1200, 40)     # Drone icon
        self.tf_health_drone = (1200, 100)     # Drone table top-left
        self.c_gv_icon       = (1200, 330)    # GV icon
        self.tf_health_gv    = (1200, 390)    # GV table top-left

        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)
        self.screen = pygame.display.set_mode((BOUND_X_MAX, BOUND_Y_MAX), 0)

        # Human panel now sits below GV panel
        self.title_human = Font(FONT, FONT_SIZE, (1200, 380))  # was 580
        self.title_human.update('   Human Status')
        self.title_human.update('ID    Utilization        Task')
        self.human_panels = [
            HumanWorkload(self.screen, (1200, 400 + i*(line_height*FONT_SIZE + 20)), human)
            for i, human in enumerate(self.humans)
        ]

        # Size transform from meter to gui
        self.ratio = 900 / 3.0
        self.center = [1125 / 2.0, 900 / 2.0]
        self.altitude = [-75.0, 200.0]

        # Game title on the window
        pygame.display.set_caption('Drone SAR Mission')

        # game clock initialization
        self.clock = pygame.time.Clock()

        ################# Main map ####################
        # Terrain
        self.background = Background(file_name=IMAGE_PATH + 'terrain_blur2.png',
                                     bound_x_min=0, bound_x_max=1125, bound_y_min=0, bound_y_max=900)
        self.map_height, self.map_width = self.background.max_bound[0], self.background.max_bound[1]
        # Awareness
        self.awareness_map = np.ones((self.map_height, self.map_width), dtype=np.float32) # Should be zeros, ones is just for testing
        # Time status
        self.status = Font(FONT, FONT_SIZE, (20, 20))
        # Targets
        self.target = []
        self.target_clicked = 0
        # self.target_decided = True
        self.task = None
        # Voronoi diagram
        self.voronoi = None
        
        # Victim flags
        n_drones = len(drones)
        self.victim_id = [0 for _ in range(n_drones)]
        self.victim_detected = [False for _ in range(n_drones)]
        self.victim_clicked = [0 for _ in range(n_drones)]
        self.victim_block_choice = [False for _ in range(n_drones)]
        self.victim_timing = [0 for _ in range(n_drones)]

        # Drones
        self.drone_images = [Vehicle(file_name=IMAGE_PATH + 'drone1.png', surface=self.screen, sc=0.06, rt=0.0) for _ in range(self.n_drones)]
        # Take-off position
        self.takeoff_position = None
        # Ground vehicles
        self.gv_images = [Vehicle(file_name=IMAGE_PATH + 'van.png', surface=self.screen, sc=0.09, rt=0.0) for _ in range(self.n_gvs)]
        # Hospital. It's not a vehicle, but not much difference
        self.hospital = Vehicle(file_name=IMAGE_PATH + 'hospital.png', surface=self.screen, sc=0.09, rt=0.0)
        ############# Main map ends ####################

        # Special regions
        self.special_regions = SpecialRegions(self.screen)

        ############# Legends ##########################
        self.legends = Background(file_name=IMAGE_PATH + 'legend.png',
                                  bound_x_min=1135, bound_x_max=1910, bound_y_min=710, bound_y_max=890)
        ################## Workspace ##################
        self.workspace = ws
        ############### Drone health ###################
        # Add title
        self.title_drone_health = Font(FONT, FONT_SIZE, (self.tf_health_drone[0], self.tf_health_drone[1] - 2 * FONT_SIZE * line_height))
        self.title_drone_health.update('    Drone condition')
        self.title_drone_health.update('Drone ID          Position               Latency             Task')
        # Small drone icon at the topleft of the drone health block
        self.drone_icon = Vehicle(file_name=IMAGE_PATH + 'drone1.png', surface=self.screen, sc=0.05, rt=0.0)
        # Drone health table
        self.health = [DroneHealth(
            self.screen,
            (self.tf_health_drone[0], self.tf_health_drone[1] + i*(line_height*FONT_SIZE + 20)),
            d,
            self.workspace
        ) for i, d in enumerate(self.drones)]
        ############## Drone health ends ################

        ###################### Ground vehicle health #####################
        # Add title
        self.title_gv_health = Font(FONT, FONT_SIZE, (self.tf_health_gv[0], self.tf_health_gv[1] - 2 * FONT_SIZE * line_height))
        self.title_gv_health.update('     Ground Vehicle condition')
        self.title_gv_health.update('    GV ID          Carrying            Position            Task')
        # Small ground vehicle icon at the topleft of the ground vehicle health block
        self.gv_icon = Vehicle(file_name=IMAGE_PATH + 'van.png', surface=self.screen, sc=0.05, rt=0.0)
        # Ground vehicle health table
        self.gv_health = [GVHealth(
            self.screen,
            (self.tf_health_gv[0], self.tf_health_gv[1] + i*(line_height*FONT_SIZE + 20)),
            g,
            self.workspace
        ) for i, g in enumerate(self.gvs)]
        ##################### Ground vehicle health ends ##################

        ###################### Wind #####################
        self.wind = []
        #################### Wind ends ##################

        ###################### Environment #####################
        # self.environment_info = EnvironmentInfo(self.screen)
        #################### Environment ends ##################

        ################## Path planning ##################
        self.paths = []
        ################ Path planning end ################

        ################ Human workload ###################
        self.humans = humans
         # Title for human workload
        self.title_human = Font(FONT, FONT_SIZE, (1200, 530))
        self.title_human.update('     Human condition')
        self.title_human.update('ID                    Utilization                      Task')
        # Create one HumanWorkload per human
        self.human_panels = [
            HumanWorkload(self.screen,
                          (1200, 600 + i*(line_height*FONT_SIZE + 20)),
                          human)
            for i, human in enumerate(self.humans)
        ]
        ############# Human workload ends #################

    def set_voronoi(self):
        print(IMAGE_PATH + 'voronoi_regions.png')
        self.vor = Background(file_name=IMAGE_PATH + 'voronoi_regions_cropped.png',
                                     bound_x_min=0, bound_x_max=1125, bound_y_min=0, bound_y_max=900)
        self.vor.surface.set_alpha(128)

    def render(self, vor, centroids):
        # Record start time
        pygame.event.get()  # Process events to avoid blocking
        if self.initial:
            self.t0 = pygame.time.get_ticks()
            self.initial = False

        # Background
        self.screen.fill(WHITE)
        self.screen.blit(self.background.surface, self.background.rect)
        # self.screen.blit(self.vor.surface, self.vor.rect)

        # ---- draw all rectangular obstacles ----
        env_obj = self.workspace.env 

        # ---- draw all rectangular obstacles with transparency and grey ----
        for (x, y, w, h) in env_obj.obs_boundary + env_obj.obs_rectangle:
            px1, py1 = self.workspace.meter_to_pixel((x,      y + h), screen_size=(1125, 900))
            px2, py2 = self.workspace.meter_to_pixel((x + w,  y    ), screen_size=(1125, 900))

            rect_width = px2 - px1
            rect_height = py2 - py1

            # Create a semi-transparent surface
            rect_surface = pygame.Surface((rect_width, rect_height), pygame.SRCALPHA)
            grey_with_alpha = (200, 200, 200, 200)  # grey + 120/255 alpha
            pygame.draw.rect(rect_surface, grey_with_alpha, rect_surface.get_rect())

            # Blit onto main screen
            self.screen.blit(rect_surface, (px1, py1))

        # ---- draw all circular obstacles with transparency and grey ----
        for (cx, cy, r) in env_obj.obs_circle:
            cx_px, cy_px = self.workspace.meter_to_pixel((cx, cy), screen_size=(1125, 900))
            xedge_px, _  = self.workspace.meter_to_pixel((cx + r, cy), screen_size=(1125, 900))
            radius_px = abs(xedge_px - cx_px)

            # Create a semi-transparent surface
            circle_surface = pygame.Surface((2*radius_px, 2*radius_px), pygame.SRCALPHA)
            grey_with_alpha = (150, 255, 255, 200)  # grey + transparency
            pygame.draw.circle(circle_surface, grey_with_alpha, (radius_px, radius_px), radius_px)

            # Blit onto main screen (centered)
            self.screen.blit(circle_surface, (cx_px - radius_px, cy_px - radius_px))

        # Draw Voronoi boundaries
        for ridge in vor.ridge_vertices:
            if -1 in ridge:
                continue  # Skip infinite ridges
            pt1 = self.workspace.grid_to_pixel(vor.vertices[ridge[0]])
            pt2 = self.workspace.grid_to_pixel(vor.vertices[ridge[1]])
            # Clip the line to the background boundary

            # Define the background boundary as a rectangle
            boundary = box(
                self.background.rect.left,
                self.background.rect.top,
                self.background.rect.right,
                self.background.rect.bottom
            )

            line = LineString([pt1, pt2])
            clipped = line.intersection(boundary)

            if clipped.is_empty:
                continue
            if clipped.geom_type == 'LineString':
                coords = list(clipped.coords)
                pygame.draw.line(self.screen, (0, 255, 0), coords[0], coords[1], 2)
            elif clipped.geom_type == 'MultiLineString':
                for seg in clipped:
                    coords = list(seg.coords)
                    pygame.draw.line(self.screen, (0, 255, 0), coords[0], coords[1], 2)

        # Draw centroids as targets
        centroids_gui = self.workspace.grid_to_pixel_array(centroids)
        for idx, centroid in enumerate(centroids_gui):
            pygame.draw.circle(self.screen, GREY, centroid.astype(int), 10)
            font = pygame.font.Font(None, 24)
            text = font.render(f'{idx+1}', True, (255,255,255))
            self.screen.blit(text, (centroid[0]-6, centroid[1]-6))

        # Legends
        self.screen.blit(self.legends.surface, self.legends.rect)

        ##################### Map ##########################
        # Awareness map
        shadow = (self.awareness_map).astype(np.uint8)
        shadow_surface = pygame.surfarray.make_surface(np.stack([shadow]*3, axis=-1))
        shadow_surface.set_alpha(128)  # semi-transparent
        self.screen.blit(shadow_surface, (0, 0))
        # Targets
        for row in (self.task or []):
            tid, pos, priority = row[0], row[1], row[2]
            self.draw_target(pos, tid, priority)
        ##################### Map ends ##########################

        ###################### Legends #####################
        # Draw grid-based base area (e.g., 3x3 at bottom-left)
        for (x, y) in self.workspace.base_area:
            px, py = self.workspace.meter_to_pixel((x, y), screen_size=(1125, 900))
            rect = pygame.Rect(px - 30, py, 22.5, 22.5)  # 22.5x22.5 = 1125/50, 900/40
            pygame.draw.rect(self.screen, (173, 216, 230), rect)  # light blue
        
        bx = sum([x for (x, y) in self.workspace.base_area]) / len(self.workspace.base_area)
        by = sum([y for (x, y) in self.workspace.base_area]) / len(self.workspace.base_area)
        px, py = self.workspace.meter_to_pixel((bx + 0.5, by + 0.5), screen_size=(1125, 900))
        pygame.draw.circle(self.screen, (0, 0, 0), (px - 30, py + 18), 15)

        # Draw hospital area and icon
        for (x, y) in self.workspace.hospital_area:
            px, py = self.workspace.meter_to_pixel((x, y), screen_size=(1125, 900))
            rect = pygame.Rect(px, py, 18, 18)  # Adjust cell size if needed
            pygame.draw.rect(self.screen, (255, 182, 193), rect)  # light pink

        hx = sum([x for (x, y) in self.workspace.hospital_area]) / len(self.workspace.hospital_area)
        hy = sum([y for (x, y) in self.workspace.hospital_area]) / len(self.workspace.hospital_area)
        hx_px, hy_px = self.workspace.meter_to_pixel((hx + 0.5, hy + 0.5), screen_size=(1125, 900))
        self.hospital.draw((hx_px, hy_px))
        ##################### Legends ends ####################

        ##################### Path planning ####################
        # Draw planned trajectories
        if hasattr(self, 'paths'):
            for color, pts in self.paths:
                # aa‐lines gives smoother curves if desired
                pygame.draw.lines(self.screen, color, False, pts, 2)
        ################### Path planning end ##################

        ##################### Agents ##########################
        # Drones
        for i, d in enumerate(self.drones):
            pos_image = tuple(self.position_meter_to_gui([d.position[0:2]]))
            height = d.position[2]
            self.drone_images[i].draw(pos_image, height)
        # GVs
        for i, g in enumerate(self.gvs):
            pos_image = tuple(self.position_meter_to_gui([g.position]))
            self.gv_images[i].draw(pos_image)
        ################### Agents ends #######################

        ###################### Drone health #####################
        for text, pos in self.title_drone_health.texts:
            self.screen.blit(text, pos)
        self.drone_icon.draw(self.c_drone_icon)
        for h in self.health:
            h.draw()
        ##################### Drone health ends ##################

        ####################### GV health #####################
        for text, pos in self.title_gv_health.texts:
            self.screen.blit(text, pos)
        self.gv_icon.draw(self.c_gv_icon)
        for h in self.gv_health:
            h.draw()
        ##################### GV health ends ####################

        ################## Human Workload #########################
        for text, pos in self.title_human.texts:
            self.screen.blit(text, pos)
        for panel in self.human_panels:
            panel.draw()
        ################## Human Workload ends #########################

        ####################### Wind ############################
        wind_gui = [self.meter_to_gui(w) for w in self.wind]
        for i, value in enumerate(wind_gui):
            wind_circle = pygame.Surface((2*value[2], 2*value[2]), pygame.SRCALPHA)
            pygame.draw.circle(wind_circle, (35, 250, 152, 100), (value[2], value[2]), value[2])
            self.screen.blit(wind_circle, (value[0] - value[2], value[1] - value[2]))
        ####################### Wind ends ########################

        ####################### Special regions ##################
        # Draw special regions
        self.special_regions.draw()
        ####################### Special regions end #############

        ####################### Environment ######################
        # self.environment_info.draw(wind_gui)
        ####################### Environment ends ##################

        ##################### Time status ##########################
        self.status.clear()
        time_display = (pygame.time.get_ticks() - self.t0) * 1e-3
        self.status.update('Time: %.1f sec' % time_display, text_color=WHITE)
        for text in self.status.texts:
            self.screen.blit(text[0], text[1])
        ######################## Time status ends ####################

        # ====== Draw panel outlines (after content, so borders stay visible) ======
        for r in (self.panel_drones,
                  self.panel_gvs,
                  self.panel_humans,
                  self.panel_legend):
            pygame.draw.rect(self.screen, BLACK, r, width=2)

        pygame.display.flip()
    
    def position_meter_to_gui(self, p_meter):
        # print("p_meter", p_meter)
        p_gui = np.array(p_meter)
        for k in range(len(p_meter)):
            p_gui[k][0] = self.ratio * p_gui[k][0] + self.center[0]
            p_gui[k][1] = -self.ratio * p_gui[k][1] + self.center[1]
        return p_gui


    def update_awareness(self, drone_positions, radius=100, increment=10.0):
        # Vectorized update of awareness map for each drone position
        yy, xx = np.ogrid[:self.map_height, :self.map_width]
        for pos in drone_positions:
            # print(drone_positions)
            # print(pos)
            cy, cx = int(pos[0]), int(pos[1])
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            mask = dist <= radius
            self.awareness_map[mask] += increment * (1 - dist[mask] / radius)
        # Ensure awareness map does not exceed a maximum value
        self.awareness_map = np.clip(self.awareness_map, 0, 255)
        # Decay awareness over time:
        self.awareness_map *= (1 - 1e-3)  # slow decay

    def set_takeoff_positions(self, position):
        self.takeoff_position = position

    def set_target(self, target=None):
        if target is not None:
            self.target = target
    
    def set_task(self, task):
        self.task = task
        # if new_target is not None:
        #     self.new_target = new_target

    def draw_target(self, pos, idx, priority):
        if priority == 0:
            color = BLUE
        elif priority == 1:
            color = ORANGE
        elif priority == 2:
            color = RED

        pygame.draw.circle(self.screen, color, pos, 10)
        font = pygame.font.Font(None, 24)
        text = font.render(f'{idx}', True, WHITE)
        self.screen.blit(text, (pos[0]-6, pos[1]-6))

    def set_wind(self, wind, meter=True):
        # if meter:
        #     wind[0] = self.ratio * wind[0] + self.center[0]
        #     wind[1] = -self.ratio * wind[1] + self.center[1]
        #     wind[2] = self.ratio * wind[2]
        self.wind.append(wind)

    def meter_to_gui(self, p_meter):
        p_return = p_meter.copy()
        p_return[0] = self.ratio * p_meter[0] + self.center[0]
        p_return[1] = -self.ratio * p_meter[1] + self.center[1]
        p_return[2] = self.ratio * p_meter[2]
        return p_return

    def reset_wind(self):
        self.wind = []  

    def workload_render(self, event=None):
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Survey: Workload', True, WHITE, BLACK)
        text_rect = text.get_rect()
        text_rect.center = (BOUND_X_MAX // 2, BOUND_Y_MAX // 3)
        self.screen.fill(BLACK)
        self.screen.blit(text, text_rect)

        # Survey buttons
        low_button = Button((360, 400, 200, 200), WHITE, 'LOW', text_color=RED, font_size=2 * FONT_SIZE)
        medium_button = Button((860, 400, 200, 200), WHITE, 'MEDIUM', text_color=RED, font_size=2 * FONT_SIZE)
        high_button = Button((1360, 400, 200, 200), WHITE, 'HIGH', text_color=RED, font_size=2 * FONT_SIZE)
        low_button.draw(self.screen)
        medium_button.draw(self.screen)
        high_button.draw(self.screen)

        # Descriptions
        # Low
        lows = [font.render('When there is sufficient', True, WHITE, BLACK),
                font.render('capacity to handle current', True, WHITE, BLACK),
                font.render('tasks comfortably with room', True, WHITE, BLACK),
                font.render('to take on additional tasks.', True, WHITE, BLACK)]
        for i, low in enumerate(lows):
            low_rect = low.get_rect()
            low_rect.topleft = (160, 700 + 35 * i)
            self.screen.blit(low, low_rect)
        # Medium
        mediums = [font.render('When there is limited spare', True, WHITE, BLACK),
                   font.render('capacity to take on a few', True, WHITE, BLACK),
                   font.render('additional tasks without significantly', True, WHITE, BLACK),
                   font.render('impacting overall performance.', True, WHITE, BLACK)]
        for i, medium in enumerate(mediums):
            medium_rect = medium.get_rect()
            medium_rect.topleft = (720, 700 + 35 * i)
            self.screen.blit(medium, medium_rect)
        # High
        highs = [font.render('When the volume of tasks meets', True, WHITE, BLACK),
                 font.render('or exceeds your capacity to', True, WHITE, BLACK),
                 font.render('respond efficiently, leaving no', True, WHITE, BLACK),
                 font.render('room for additional tasks.', True, WHITE, BLACK)]
        for i, high in enumerate(highs):
            high_rect = high.get_rect()
            high_rect.topleft = (1360, 700 + 35 * i)
            self.screen.blit(high, high_rect)

        # Update
        pygame.display.flip()
        if any([low_button.handle_event(event), medium_button.handle_event(event), high_button.handle_event(event)]):
            return True
        else:
            return False

        

    def perceived_risk_render(self, event=None):
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Survey: Perceived-Risk', True, WHITE, (50, 50, 50))
        text_rect = text.get_rect()
        text_rect.center = (BOUND_X_MAX // 2, BOUND_Y_MAX // 3)
        self.screen.fill((50, 50, 50))
        self.screen.blit(text, text_rect)

        # Survey buttons
        low_button = Button((360, 400, 200, 200), WHITE, 'LOW', text_color=BLUE, font_size=2 * FONT_SIZE)
        medium_button = Button((860, 400, 200, 200), WHITE, 'MEDIUM', text_color=BLUE, font_size=2 * FONT_SIZE)
        high_button = Button((1360, 400, 200, 200), WHITE, 'HIGH', text_color=BLUE, font_size=2 * FONT_SIZE)
        low_button.draw(self.screen)
        medium_button.draw(self.screen)
        high_button.draw(self.screen)

        # Description
        definitions = [font.render('During the mission, how high', True, WHITE, BLACK),
                       font.render('was the risk of misidentifying humans', True, WHITE, BLACK),
                       font.render('in need of assistance and', True, WHITE, BLACK),
                       font.render('misinforming the command center?', True, WHITE, BLACK)]
        for i, line in enumerate(definitions):
            line_rect = line.get_rect()
            line_rect.topleft = (720, 700 + 35 * i)
            self.screen.blit(line, line_rect)

        pygame.display.flip()

        if any([low_button.handle_event(event), medium_button.handle_event(event), high_button.handle_event(event)]):
            return True
        else:
            return False


if __name__ == "__main__":
    pygame.init()
    drones = [VirtualDrone(0, (-1.2, -0.5)), VirtualDrone(1, (-1.2, 0.5))]
    gvs = [VirtualGV(0, (-1.2, -1)), VirtualGV(0, (-1.2, 1))]
    takeoff_positions = [d.position[0:2] for d in drones]
    game_mgr = GameMgr(drones, gvs)
    game_mgr.set_takeoff_positions(takeoff_positions)
    # game_mgr.set_wind([0, 0, 0.05])
    game_mgr.set_wind([0, 1, 0.05])
    priority = []
    tasks = []
    target_remaining = [[0, 0], [-1.0, 0], [1.0, 0]]
    for idx, target in enumerate(target_remaining):
        target = game_mgr.position_meter_to_gui([target])[0]
        tasks.append([idx + 1, target, 0])

    game_mgr.set_task(tasks)
    running = True
    while running:
        # Update drone positions and health
        for d in drones:
            d.health -= 0.1
            d.position[2] += 0.1
            d.position[0] += 0.001

        for g in gvs:
            g.position[0] += 0.001
            

        pos = [(d.position[0], d.position[1]) for d in drones]
        # print(pos)
        # print()
        pos = game_mgr.position_meter_to_gui(pos)
        # print("Drone positions:", pos)
        # assert False
        game_mgr.update_awareness(pos, radius=0)
        game_mgr.render()

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Limit frame rate
        game_mgr.clock.tick(60)