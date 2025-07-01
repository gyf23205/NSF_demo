import pygame
import os
from examples.vehicles import *
from constants import *
from util_classes import Font, Button, Bar
from pygame.font import SysFont


class DroneHealth:
    def __init__(self, screen, pos, virtual_drone):
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
        # self.status_txt = SysFont(FONT, FONT_SIZE)

    def draw(self):
        idx_txt = self.idx_txt.render('         ' + str(self.drone.idx), True, BLACK)
        self.screen.blit(idx_txt, (self.x0, self.y0, self.grid_width, self.grid_height))
        self.alt_bar.draw(self.drone.position[2])
        self.health_bar.draw(self.drone.health)
        pos_str = f"({self.drone.position[0]:.2f}, {self.drone.position[1]:.2f})"
        pos_txt = self.current_pos_txt.render(pos_str, True, BLACK)
        self.screen.blit(pos_txt, (self.x0 + 3*(self.grid_width+self.spacing), self.y0, self.grid_width, self.grid_height))
        # status_txt = self.status_txt.render(...)
        # self.screen.blit(pos_txt, (self.x0 + 4*self.grid_width, self.y0, self.grid_width, self.grid_height))

    # def update_health(self, reduction):
    #     self.health = max(0, self.health - reduction)
    #     return self.health

    # def is_healthy(self):
    #     return self.health > 0


class Background:
    def __init__(self, file_name, bound_x_min, bound_x_max, bound_y_min, bound_y_max):
        self.figure = pygame.image.load(file_name).convert()
        self.surface = self.figure
        self.surface = pygame.transform.scale(self.surface, (bound_x_max, bound_y_max))
        self.rect = self.surface.get_rect()
        self.min_bound = np.array([bound_x_min, bound_y_min])
        self.max_bound = np.array([bound_x_max, bound_y_max])


class GameMgr:
    def __init__(self, drones, gvs):
        pygame.init()
        self.t0 = 0
        self.initial = True
        self.drones = drones
        self.gvs = gvs
        self.n_drones = len(drones)
        
        # Define position of blocks, tf=topleft, c=center
        self.tf_map = (100, 100)
        self.tf_health = (950, 120)
        self.c_drone_icon = (950, 70)

        # Make sure to call pygame.init() before

        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)
        self.screen = pygame.display.set_mode((BOUND_X_MAX, BOUND_Y_MAX), 0)

        # Size transform from meter to gui
        self.ratio = 240.0
        self.center = [450.0, 360.0]
        self.altitude = [-75.0, 200.0]

        # Game title on the window
        pygame.display.set_caption('Drone SAR Mission')

        # game clock initialization
        self.clock = pygame.time.Clock()

        ################# Main map ####################
        # Terrain
        self.background = Background(file_name=IMAGE_PATH + 'terrain_blur.png',
                                     bound_x_min=0, bound_x_max=900, bound_y_min=0, bound_y_max=720)
        self.map_height, self.map_width = self.background.max_bound[0], self.background.max_bound[1]
        # Awareness
        self.awareness_map = np.ones((self.map_height, self.map_width), dtype=np.float32) # Should be zeros, ones is just for testing
        # Time status
        self.status = Font(FONT, FONT_SIZE, (20, 200))
        # Targets
        self.target = []
        self.target_clicked = 0
        self.target_decided = True
        # self.new_target = []
        # self.new_target_triggered = False
        # Drones
        self.drone_images = [Vehicle(file_name=IMAGE_PATH + 'drone1.png', surface=self.screen, sc=0.07, rt=0.0) for _ in range(self.n_drones)]
        # Take-off position
        self.takeoff_position = None
        # Groung vehicles
        self.gv_images = [Vehicle(file_name=IMAGE_PATH + 'van.png', surface=self.screen, sc=0.09, rt=0.0) for _ in range(self.n_drones)]
        ############# Main map ends ####################

        ############### Drone health ###################
        # Small drone icon at the topleft of the drone health block
        self.drone_icon = Vehicle(file_name=IMAGE_PATH + 'drone1.png', surface=self.screen, sc=0.05, rt=0.0)

        # Drone health table
        self.health = [DroneHealth(self.screen, (self.tf_health[0], self.tf_health[1] + i*(line_height*FONT_SIZE + 20)), d) for i, d in enumerate(self.drones)]
        ############## Drone health ends ################

        ###################### Wind #####################
        self.wind = []
        self.wind_danger = False
        self.wind_triggered = False
        self.wind_decided = True
        self.wind_closed = True
        self.wind_clicked = 0
        #################### Wind ends ##################

        ###################### Mission #####################
        #################### Mission ends ##################

    def render(self):
        # Record start time
        if self.initial:
            self.t0 = pygame.time.get_ticks()
            self.initial = False

        # Background
        self.screen.fill(WHITE)
        self.screen.blit(self.background.surface, self.background.rect)

        ##################### Map ##########################
        # Awareness map
        shadow = (self.awareness_map).astype(np.uint8)
        shadow_surface = pygame.surfarray.make_surface(np.stack([shadow]*3, axis=-1))
        shadow_surface.set_alpha(128)  # semi-transparent
        self.screen.blit(shadow_surface, (0, 0))
        # Drones
        for i, d in enumerate(self.drones):
            pos_image = tuple(self.position_meter_to_gui([d.position[0:2]]))
            self.drone_images[i].draw(pos_image)
        # GVs
        for i, g in enumerate(self.gvs):
            pos_image = tuple(self.position_meter_to_gui([g.position]))
            self.gv_images[i].draw(pos_image)
        # Targets
        assert len(priority) == len(self.target)
        for i, pos in enumerate(self.target):
            pygame.draw.circle(self.screen, BLUE if priority[i] <= 0 else RED, pos, 10)
        # for i, pos in enumerate(self.new_target):
        #     pygame.draw.circle(self.screen, RED, pos, 15)
        # Takeoff positions
        for i, pos in enumerate(self.takeoff_position):
            pygame.draw.circle(self.screen, BLACK, pos, 10)
        ##################### Map ends ##########################

        ###################### Drone health #####################
        self.drone_icon.draw(self.c_drone_icon)
        for h in self.health:
            h.draw()
        ##################### Drone health ends ##################

        ####################### Wind ############################
        for i, value in enumerate(self.wind):
            pygame.draw.circle(self.screen, (35, 250, 152), [value[0], value[1]], value[2])
        ####################### Wind ends ########################
        pygame.display.flip()
    
    def position_meter_to_gui(self, p_meter):
        p_gui = np.array(p_meter)
        for k in range(len(p_meter)):
            p_gui[k][0] = self.ratio * p_gui[k][0] + self.center[0]
            p_gui[k][1] = -self.ratio * p_gui[k][1] + self.center[1]
        return p_gui


    def update_awareness(self, drone_positions, radius=100, increment=10.0):
        # Vectorized update of awareness map for each drone position
        yy, xx = np.ogrid[:self.map_height, :self.map_width]
        for pos in drone_positions:
            print(pos)
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
        # if new_target is not None:
        #     self.new_target = new_target

    def set_wind(self, wind, meter=True):
        wind_copy = wind.copy()
        if meter:
            wind_copy[0] = self.ratio * wind[0] + self.center[0]
            wind_copy[1] = -self.ratio * wind[1] + self.center[1]
            wind_copy[2] = self.ratio * wind[2]
        self.wind.append(wind_copy)

    def reset_wind(self):
        self.wind = []




if __name__ == "__main__":
    pygame.init()
    drones = [VirtualDrone(0, (-1.2, -0.5)), VirtualDrone(1, (-1.2, 0.5))]
    gvs = [VirtualGV(0, (-1.2, -1)), VirtualGV(0, (-1.2, 1))]
    takeoff_positions = [d.position[0:2] for d in drones]
    game_mgr = GameMgr(drones, gvs)
    game_mgr.set_takeoff_positions(takeoff_positions)
    game_mgr.set_wind([0, 0, 0.05])
    game_mgr.set_wind([0, 1, 0.05])
    priority = []

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
        game_mgr.update_awareness(game_mgr.position_meter_to_gui(pos), radius=70)
        game_mgr.render()

        # Check for quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Limit frame rate
        game_mgr.clock.tick(60)