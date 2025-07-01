import pygame
from PIL import Image
import os
from util_classes import Font, Button, TextInput
import multiprocessing as mp
import socket
import json
from constants import *


class Task:
    def __init__(self, surface, task_id, target_loc, task_pos, priority=0):
        self.task_id = task_id
        self.surface = surface
        self.x0, self.y0 = task_pos
        self.target_pos = target_loc
        self.priority = priority
        self.assigned_drone = None
        self.assigned_gv = None
        self.assigned_time = pygame.time.get_ticks()
        self.reject_time_limit = 5000
        self.reject = False

        self.grid_width = 100
        self.grid_height = line_height * FONT_SIZE
        self.task_id_text = Font(FONT, FONT_SIZE, (self.x0, self.y0))
        self.target_pos_text = Font(FONT, FONT_SIZE, (self.x0 + self.grid_width, self.y0))
        self.priority_input = TextInput((self.x0 + 2 * self.grid_width, self.y0,self.grid_width, self.grid_height), color=WHITE, maximum=2)
        self.assigned_drone_text = Font(FONT, FONT_SIZE, (self.x0 + 3 * self.grid_width, self.y0))
        self.assigned_gv_text = Font(FONT, FONT_SIZE, (self.x0 + 4 * self.grid_width, self.y0))
        self.rejection_button = Button((self.x0 + 5 * self.grid_width, self.y0, self.grid_width, self.grid_height), RED, "Reject", text_color=WHITE)
        # self.assigned_gv_input = TextInput((self.x0 + 3 * self.grid_width, self.y0, self.grid_width, self.grid_height), color=WHITE, maximum=n_gvs)

        self.task_id_text.update(f'{self.task_id}')
        self.target_pos_text.update(f'{self.target_pos}')
        self.priority_input.text = str(self.priority)
        self.assigned_drone_text.update(str(self.assigned_drone))
        self.assigned_gv_text.update(str(self.assigned_gv))

        # self.assigned_gv_input.text = str(self.assigned_gv)

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
        self.priority_input.draw(self.surface)
        for text in self.assigned_drone_text.texts:
            self.surface.blit(text[0], text[1])
        for text in self.assigned_gv_text.texts:
            self.surface.blit(text[0], text[1])

        current_time = pygame.time.get_ticks()
        if current_time - self.assigned_time < self.reject_time_limit:
            self.rejection_button.draw(self.surface)
        else:
            # Clear the area where the rejection button would be drawn
            pygame.draw.rect(self.surface, WHITE, self.rejection_button.rect)
            
        # self.assigned_gv_input.draw(self.surface)

    def handle_event(self, event):
        old_priority = self.priority_input.text
        old_reject = self.reject
        self.priority_input.handle_event(event)
        current_time = pygame.time.get_ticks()
        if current_time - self.assigned_time < self.reject_time_limit:
            self.reject = self.rejection_button.handle_event(event)
        if (old_priority != self.priority_input.text) or (old_reject != self.reject):
            return True
        return False
        # self.assigned_gv_input.handle_event(event)


class Human:
    def __init__(self, idx):
        self.idx = idx
        self.progress = 0.0
        self.workload = 'low'

class UserGUI:
    def __init__(self):
        pygame.init()
        self.screen_width = 1200
        self.screen_height = 750
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill(WHITE)
        

        # Victim block
        self.image_width = 400
        self.image_height = 280
        self.image_rect = pygame.Rect(
            70,
            200,
            self.image_width,
            self.image_height
        )
        # Center the buttons below the image
        button_width = 120
        button_height = 50
        spacing = 50
        buttons_y = self.image_rect.bottom + 10
        # total_buttons_width = button_width * 2 + spacing
        buttons_x = self.image_rect.x + (0.5*self.image_width - button_width - 0.5*spacing)
        accept_rect = pygame.Rect(buttons_x, buttons_y, button_width, button_height)
        reject_rect = pygame.Rect(buttons_x + button_width + spacing, buttons_y, button_width, button_height)
        self.button_accept =  Button(accept_rect, BLUE, 'Accept')
        self.button_reject = Button(reject_rect, RED, 'Reject')
        self.image = None
        self.button_accept.draw(self.screen)
        self.button_reject.draw(self.screen)

        # Workload block
        self.workload_text = Font(FONT, FONT_SIZE, (70, 70))
        self.workload = 'low'
        self.workload_text.update('Workload: '+ self.workload)
        self.screen.blit(self.workload_text.texts[0][0], self.workload_text.texts[0][1])
        
        # weather block
        weather_x = 600
        weather_y = 150
        self.weather_text = Font(FONT, FONT_SIZE, (weather_x, weather_y))
        self.weather = 'sunny'
        self.weather_text.update('Weather: '+ self.weather)
        self.button_wind_change = Button((weather_x, weather_y + FONT_SIZE * line_height + 30, button_width, button_height), (0, 255, 0), "Change routes")
        self.button_wind_maintain = Button((weather_x + button_width + spacing, weather_y + FONT_SIZE * line_height + 30, button_width, button_height), YELLOW, "Maintain routes")
        self.button_wind_change.draw(self.screen)
        self.button_wind_maintain.draw(self.screen)

        # Task block
        taks_x = 550
        task_y = 400
        # self.received_new_tasks = False
        self.task_text = Font(FONT, FONT_SIZE, (taks_x, task_y))
        self.task_text.update('                       Task Monitor')
        self.task_text.update('Task ID        Target pos        Priority       Assigned Drone      Assigned GV')
        for text in self.task_text.texts:
            self.screen.blit(text[0], text[1])
        self.task_list_x = taks_x
        self.task_list_y = task_y + len(self.task_text.texts) * line_height * FONT_SIZE
        self.task_list = []


    def render(self):
        # self.screen.fill(WHITE)

        # Update workload text
        if data and data['workload'] is not None:
            self.workload_text.clear()
            self.workload_text.update('Workload: ' + data['workload'])

        # Victim block
        if data and data['idx_image'] is not None:
            # print(data['idx_image'])
            image_path = f"examples/images/victim{data['idx_image']}.jpeg"
            pil_image = Image.open(image_path)
            pil_image = pil_image.resize((self.image_width, self.image_height))
            image = pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode)
            if image is not None:
                self.image = image
        if self.image is not None:
            self.screen.blit(self.image, self.image_rect)
        else:
            pygame.draw.rect(self.screen, WHITE, self.image_rect)
            

        # Task block
        if self.task_list:
            for i, task in enumerate(self.task_list): # tasks is defined in main
                task_pos = (self.task_list_x, self.task_list_y + i * FONT_SIZE * line_height)
                task_id, target_loc, priority = task
                new_task = Task(self.screen, task_id, target_loc, task_pos, priority)
                self.task_list.append(new_task)
                print(new_task.priority_input.text)
        if self.task_list:
            for task in self.task_list:
                task.draw()

        # weather block
        if data and data['wind'] is not None:
            self.weather = data['wind']
            self.weather_text.clear()
            self.weather_text.update('Weather: ' + str(self.weather))
            self.screen.blit(self.weather_text.texts[0][0], self.weather_text.texts[0][1])

        pygame.display.flip()

# Example usage:
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEO_WINDOW_POS'] = "800,100"
    host = '127.0.0.1'  # IP of the server
    port = 8888
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.setblocking(False)
    # tasks = [(1, (100, 200), 0), (2, (300, 400), 1)]  # Example tasks
    # workload = 'low'  # Example workload

    gui = UserGUI()

    # response['victim']: b'reject' or b'accept'
    # response['weather_decision']: 'change' or 'maintain'
    # response['tasks']: list of Task objects
    data = None # Data received from the server
    response = {'victim': None, 'weather_decision': None, 'tasks': None} # Response to be sent back to the server
    running = True
    while running:
        response_changed = False
        # Receive weather, task, victim from server
        try:
            data_received = s.recv(1024).decode() 
            if data_received:
                data = json.loads(data_received)
        except BlockingIOError:
            pass


        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Victim handling
            if gui.button_accept.handle_event(event):
                gui.image = None
                data['idx_image'] = None
                response_changed = True
                response['victim'] = b'accept'
            elif gui.button_reject.handle_event(event):
                gui.image = None
                data['idx_image'] = None
                response_changed = True
                response['victim'] = b'reject'
            else:
                pass

            ########## Humane handling ##############
            ########## Human handling ends ##########


            # Task handling
            tasks_changed = False
            for task in gui.task_list:
                if task.handle_event(event):
                    tasks_changed = True
            if tasks_changed:
                response_changed = True
                response['tasks'] = gui.task_list

            # Weather handling
            if gui.button_wind_change.handle_event(event):
                response_changed = True
                response['weather_decision'] = 'change'
            elif gui.button_wind_maintain.handle_event(event):
                response_changed = True
                response['weather_decision'] = 'maintain'
            else:
                pass
            

        # Render the GUI and get the response        
        gui.render()
        idx_image = None  # Reset image after rendering

        # Send response back to the server if it has changed
        if response_changed:
            print('Sending response to server')
            msg = json.dumps(response).encode('utf-8')
            s.sendall(msg)
    pygame.quit()
