import pygame
from PIL import Image
import os
from util_classes import Font, Button, Task
import multiprocessing as mp
import socket
import json
from constants import *

class UserGUI:
    def __init__(self):
        pygame.init()
        self.screen_width = 1000
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
        self.task_text = Font(FONT, FONT_SIZE, (taks_x, task_y))
        self.task_text.update('                          Task Monitor')
        self.task_text.update('Task ID    Target pos    Assigned drone    Assigned GV ')
        for text in self.task_text.texts:
            self.screen.blit(text[0], text[1])
        self.task_list_x = taks_x
        self.task_list_y = task_y + len(self.task_text.texts) * line_height * FONT_SIZE
        self.task_list = []


    def render(self):
        # self.screen.fill(WHITE)

        # Update workload text
        if workload is not None:
            self.workload_text.clear()
            self.workload_text.update('Workload: ' + workload)

        # Victim block
        if data['idx_image'] is not None:
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
        if self.task_list is not None:
            for i, task in enumerate(tasks):
                task_pos = (self.task_list_x, self.task_list_y + i * FONT_SIZE * line_height)
                task_id, target_loc, assigned_drone, assigned_gv = task
                new_task = Task(self.screen, task_id, target_loc, task_pos, assigned_drone, assigned_gv)
                self.task_list.append(new_task)
        if self.task_list:
            for task in self.task_list:
                task.draw()

        # weather block
        if data['weather'] is not None:
            self.weather = data['weather']
            self.weather_text.clear()
            self.weather_text.update('Weather: ' + self.weather)
            self.screen.blit(self.weather_text.texts[0][0], self.weather_text.texts[0][1])

        pygame.display.flip()

# Example usage:
if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEO_WINDOW_POS'] = "800,100"
    # host = '127.0.0.1'  # IP of the server
    # port = 8888
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect((host, port))
    # s.setblocking(False)
    tasks = [(1, (100, 200), 0, 0), (2, (300, 400), 1, 1)]  # Example tasks
    workload = 'low'  # Example workload

    gui = UserGUI()
    data = {'idx_image': 1, 'tasks': tasks, 'weather': 'sunny'}
    response = {'victim': None, 'weather_decision': None, 'task decision': None}
    running = True
    while running:
        response_changed = False
        # # Receive weather, task, victim from server
        # try:
        #     data_received = s.recv(1024).decode() 
        #     if data_received:
        #         data = json.loads(data_received)
        # except BlockingIOError:
        #     pass


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

            # Task handling
            for task in gui.task_list:
                task.handle_event(event)

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
            # msg = json.dumps(response).encode('utf-8')
            # s.sendall(msg)
    pygame.quit()
