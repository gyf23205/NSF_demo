import pygame
from PIL import Image
import os
from util_classes import Font, Button, TextInput, TextInputResponse
import socket
import json
from constants import *

# for estimator
import csv
from TF_raw import TransformerRawClassifier
import torch
# import hydra
import json
import numpy as np
import yaml


csv_path = 'D:\\Projects\\qualisys_drone_sdk\\dummy_log\\aggregated_output.csv'


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
        # self.reject_time_limit = 100000
        # self.reject = False
        self.spacing = spacing
        self.priority_set = [0, 1, 2]

        self.grid_width = grid_width
        self.grid_height = line_height * FONT_SIZE
        self.task_id_text = Font(FONT, FONT_SIZE, (self.x0, self.y0))
        self.target_pos_text = Font(FONT, FONT_SIZE, (self.x0 + (self.grid_width + self.spacing), self.y0))
        self.priority_input = TextInput((self.x0 + 2 * (self.grid_width + self.spacing), self.y0,self.grid_width, self.grid_height), color=WHITE, maximum=max(self.priority_set))
        self.assigned_drone_text = Font(FONT, FONT_SIZE, (self.x0 + 3 * (self.grid_width + self.spacing), self.y0))
        self.assigned_gv_text = Font(FONT, FONT_SIZE, (self.x0 + 4 * (self.grid_width + self.spacing), self.y0))
        # self.rejection_button = Button((self.x0 + 5 * (self.grid_width + self.spacing), self.y0, self.grid_width, self.grid_height), RED, "Reject", text_color=WHITE)
        # self.assigned_gv_input = TextInput((self.x0 + 3 * self.grid_width, self.y0, self.grid_width, self.grid_height), color=WHITE, maximum=n_gvs)
        
        self.task_id_text.update('    ' + f'{self.task_id}')
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
        
    def reposition(self, task_pos):
        self.x0, self.y0 = task_pos
        self.task_id_text.pos = (self.x0, self.y0)
        self.target_pos_text.pos = (self.x0 + (self.grid_width + self.spacing), self.y0)
        self.priority_input.rect.topleft = (self.x0 + 2 * (self.grid_width + self.spacing), self.y0)
        self.assigned_drone_text.pos = (self.x0 + 3 * (self.grid_width + self.spacing), self.y0)
        self.assigned_gv_text.pos = (self.x0 + 4 * (self.grid_width + self.spacing), self.y0)
        # self.rejection_button.rect.topleft = (self.x0 + 5 * (self.grid_width + self.spacing), self.y0)

    def draw(self):
        for text in self.task_id_text.texts:
            self.surface.blit(text[0], text[1])
        for text in self.target_pos_text.texts:
            # text[0] = '[{:.2f}'.format(float(text[0][0])) + ', {:.2f}]'.format(float(text[0][1]))
            self.surface.blit(text[0], text[1])
        self.priority_input.draw(self.surface)

        self.assigned_drone_text.clear()
        self.assigned_drone_text.update(str(self.assigned_drone))
        # print(f'Assigned drone: {self.assigned_drone}')
        for text in self.assigned_drone_text.texts:
            self.surface.blit(text[0], text[1])

        self.assigned_gv_text.clear()
        self.assigned_gv_text.update(str(self.assigned_gv))
        for text in self.assigned_gv_text.texts:
            self.surface.blit(text[0], text[1])

        current_time = pygame.time.get_ticks()
        # if (current_time - self.assigned_time < self.reject_time_limit) and not self.reject:
        #     self.rejection_button.draw(self.surface)
            
        # self.assigned_gv_input.draw(self.surface)

    def handle_event(self, event):
        old_priority = self.priority
        if self.priority_input.text.isdigit() and int(self.priority_input.text) in self.priority_set: 
            self.priority = int(self.priority_input.text)

        # old_reject = self.reject
        self.priority_input.handle_event(event)
        current_time = pygame.time.get_ticks()

        # if current_time - self.assigned_time < self.reject_time_limit:
        #     self.reject = self.rejection_button.handle_event(event) 

        # if (old_priority != self.priority) or (old_reject != self.reject):
        if (old_priority != self.priority):
            return True
        return False
        # self.assigned_gv_input.handle_event(event)

    def to_dict(self):
        return {
            'task_id': self.task_id,
            'priority': self.priority,
            # 'reject': self.reject
        }


class Human:
    def __init__(self, idx):
        self.idx = idx
        self.progress = 0.0
        self.workload = 'low'

class UserGUI:
    def __init__(self):
        pygame.init()
        self.screen_width = 1300
        self.screen_height = 930
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill(WHITE)

        # Victim block
        self.image_width = 400
        self.image_height = 280
        self.image_rect = pygame.Rect(
            130,
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
        buttons_x = self.image_rect.x + (0.5*self.image_width - button_width - 0.5*spacing) - 120
        accept_rect = pygame.Rect(buttons_x, buttons_y, button_width, button_height)
        reject_rect = pygame.Rect(buttons_x + button_width + spacing, buttons_y, button_width, button_height)
        handover_rect = pygame.Rect(buttons_x + 2 * (button_width + spacing), buttons_y, button_width, button_height)
        self.button_accept =  Button(accept_rect, BLUE, 'Accept', text_color=WHITE)
        self.button_reject = Button(reject_rect, RED, 'Reject', text_color=WHITE)
        self.button_handover = Button(handover_rect, GREEN, 'Hand over', text_color=WHITE)
        self.image = None
        self.button_accept.draw(self.screen)
        self.button_reject.draw(self.screen)
        self.button_handover.draw(self.screen)

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
        self.button_wind_change = Button((weather_x, weather_y + FONT_SIZE * line_height + 30, button_width, button_height), BLUE, "Change routes")
        self.button_wind_maintain = Button((weather_x + button_width + spacing, weather_y + FONT_SIZE * line_height + 30, button_width, button_height), RED, "Maintain routes")
        self.button_wind_handover = Button((weather_x + 2 * (button_width + spacing), weather_y + FONT_SIZE * line_height + 30, button_width, button_height), GREEN, "Hand over")
        self.button_wind_change.draw(self.screen)
        self.button_wind_maintain.draw(self.screen)
        self.button_wind_handover.draw(self.screen)

        # Task block
        self.task_x = 550
        self.task_y = 400
        # self.received_new_tasks = False
        self.task_text = Font(FONT, FONT_SIZE, (self.task_x, self.task_y))
        self.task_text.update('                                                Task Monitor')
        self.task_text.update('Task ID                 Target pos              Priority                Assigned Drone       Assigned GV')
        for text in self.task_text.texts:
            self.screen.blit(text[0], text[1])
        self.task_list_x = self.task_x
        self.task_list_y = self.task_y + len(self.task_text.texts) * line_height * FONT_SIZE
        self.task_list = []
        self.n_previous_tasks = len(self.task_list)

        # Response block
        response_x = 40
        response_y = 650
        self.response_title = Font(FONT, FONT_SIZE, (response_x, response_y))
        self.response_title.update('Messages from victims')
        self.screen.blit(self.response_title.texts[0][0], self.response_title.texts[0][1])
        self.response_text = Font(FONT, FONT_SIZE, (response_x, response_y + FONT_SIZE * line_height))
        self.response_input = TextInputResponse((response_x, response_y + 2 * FONT_SIZE * line_height, 400, FONT_SIZE * line_height), color=WHITE, maximum=1000)

    def render(self):
        # self.screen.fill(WHITE)

        ###################### Update workload text ######################
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            last_row = rows[-1]
            last_row = list(map(float, last_row))

        # with open(csv_path, 'w', newline='') as f:
        #     f.truncate()

        # 2. run estimator model
        with open('config_ecg_gaze.yaml', 'r') as yf:
            cfg = yaml.safe_load(yf)

        model = TransformerRawClassifier(
            config=cfg["config_tf"],
            optim_cfg=cfg["optim"],
            pre_process=cfg.get("pre_process", None)
        )
        state_dict = torch.load('last_new.pt', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        ecg = last_row[:130]
        # gaze = last_row[130:]
        # gaze_au_matrix = rows[130:].reshape(10, 30)
        gaze_au_matrix = np.array(last_row[130:]).reshape(10, 30)

        t1 = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0) # raw ECG
        # t2 = torch.tensor(gaze, dtype=torch.float32).unsqueeze(0) # raw Gaze
        t2 = torch.tensor(gaze_au_matrix, dtype=torch.float32)  # [10, 30]

        if torch.isnan(t2).any() or torch.isinf(t2).any():
            print("⚠️ NaN or Inf detected in t2 (gaze input)")

        with torch.no_grad():
            out = model(t1, t2)
            pred_label = torch.argmax(out).item()
            # print(out, pred_label)
      

        # 3. update workload
        if pred_label == 1:
            workload_text = 'high'
        elif pred_label == 0:
            workload_text = 'low'      

        workload_text = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.4, 0.3])

        self.workload_text.clear()
        self.workload_text.update('Workload: ' + workload_text)  
        area = self.workload_text.rect.copy()
        area.width += 100  # Adjust width to fit the screen
        pygame.draw.rect(self.screen, WHITE, area)
        self.screen.blit(self.workload_text.texts[0][0], self.workload_text.texts[0][1])             
        ###################### Update workload text ends #####################


        # ###################### Update workload text ######################
        # if data and data['workload'] is not None:
        #     self.workload_text.clear()
        #     self.workload_text.update('Workload: ' + data['workload'])
        #     # clean the previous workload text
        #     pygame.draw.rect(self.screen, WHITE, self.workload_text.rect)
        #     self.screen.blit(self.workload_text.texts[0][0], self.workload_text.texts[0][1])
        # ###################### Update workload text ends #####################


        ###################### Victim block ######################
        if data and data['idx_image'] is not None:
            # print(data['idx_image'])
            victim_buffer.append(data['idx_image'])
            data['idx_image'] = None
        if victim_buffer:
            image_path = f"examples/images/victim{victim_buffer[0]}.jpeg"
            pil_image = Image.open(image_path)
            pil_image = pil_image.resize((self.image_width, self.image_height))
            image = pygame.image.fromstring(pil_image.tobytes(), pil_image.size, pil_image.mode)
            if image is not None:
                self.image = image
        if self.image is not None:
            self.screen.blit(self.image, self.image_rect)
        else:
            pygame.draw.rect(self.screen, WHITE, self.image_rect)
        ###################### Victim block ends ######################


        ###################### Task block ######################
        # print('data before receiving tasks: ', data)
        if data is not None and data['tasks'] is not None:
            # print('Received tasks from server: ', data['tasks'])
            self.task_list = []
            for i, task in enumerate(data['tasks']):
                task_pos = (self.task_list_x, self.task_list_y + i * FONT_SIZE * line_height)
                task_id, target_loc, priority, assigned_drone = task
                new_task = Task(self.screen, task_id, target_loc, task_pos, priority)
                new_task.assigned_drone = assigned_drone
                # print(assigned_drone)
                self.task_list.append(new_task)
                # print(new_task.priority_input.text)

        # Clear the task area before drawing
        task_table_width = 6 * (grid_width + spacing)
        task_table_height =  self.screen_height - self.task_y
        # print(f'Number of previous tasks: {self.n_previous_tasks}')
        pygame.draw.rect(self.screen, WHITE, (self.task_x, self.task_y + 2 * line_height * FONT_SIZE, task_table_width, task_table_height))
        
        if self.task_list:
            # assert False
            if self.n_previous_tasks != len(self.task_list):
                task_list_temp = []
                # Reposition remaining tasks to the top of the task region
                for i, task in enumerate(self.task_list):
                    task_pos = (self.task_list_x, self.task_list_y + i * FONT_SIZE * line_height)
                    task_id, target_loc, priority, assigned_drone = task.task_id, task.target_pos, task.priority, task.assigned_drone
                    new_task = Task(self.screen, task_id, target_loc, task_pos, priority)
                    new_task.assigned_drone = assigned_drone
                    task_list_temp.append(new_task)
                self.task_list = task_list_temp
                self.n_previous_tasks = len(self.task_list)

            # Now draw the updated task list
            for task in self.task_list:
                task.draw()
        ########################## Task block ends ######################

        ###################### Weather block ######################
        if data and data['wind_speed'] is not None:
            self.weather = data['wind_speed']
            pygame.draw.rect(self.screen, WHITE, self.weather_text.rect)  # Clear the previous weather text
            self.weather_text.clear()
            self.weather_text.update('Wind speed changes significantly, current speed: ' + "{:.2f}".format(self.weather))
            self.screen.blit(self.weather_text.texts[0][0], self.weather_text.texts[0][1])
        ###################### Weather block ends ######################

        ####################### Response block ##########################
        response_region_width = 520  # Adjust as needed
        response_region_height = 3 * FONT_SIZE * line_height + 20
        pygame.draw.rect(self.screen, WHITE, (40, 650, response_region_width, response_region_height))
        # Draw the response title every frame
        for text in self.response_title.texts:
            self.screen.blit(text[0], text[1])

        if data and data['vic_msg'] is not None:
            vic_msg_buffer.append(data['vic_msg'])
            data['vic_msg'] = None
        
        self.response_text.clear()
        if vic_msg_buffer:
            self.response_text.update(vic_msg_buffer[0])
            self.screen.blit(self.response_text.texts[0][0], self.response_text.texts[0][1])
        self.response_input.draw(self.screen)
        ###################### Response block ends ######################
        pygame.display.flip()


if __name__ == '__main__':
    import os
    os.environ['SDL_VIDEO_WINDOW_POS'] = "600,100"
    host = '127.0.0.1'  # IP of the server (localhost)
    port = 8888
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.setblocking(False)
    # tasks = [(1, (100, 200), 0), (2, (300, 400), 1)]  # Example tasks
    # workload = 'low'  # Example workload

    gui = UserGUI()

    # response['victim']: 'reject' or 'accept'
    # response['weather_decision']: 'change' or 'maintain'
    # response['tasks']: list of Task objects
    response = {'victim': None, 'weather_decision': None, 'tasks': None} # Response to be sent back to the server
    running = True
    victim_buffer = []  # Buffer to store victims
    vic_msg_buffer = []  # Buffer to store messages from victims
    data = {'idx_image': None, 'tasks': None, 'wind_speed': None, 'workload': None, 'vic_msg': None}  # Initialize data
    try:
        recv_buffer = ''
        while running:
            response_changed = False
            # Receive weather, task, victim from server
            try:
                data_received = s.recv(4096).decode()
                if data_received:
                    recv_buffer += data_received
                    while '\n' in recv_buffer:
                        line, recv_buffer = recv_buffer.split('\n', 1)
                        if line.strip():
                            data_temp = json.loads(line)
                            # print('Received data from server:', repr(data_temp))
                            for key, value in data_temp.items():
                                if value is not None:
                                    # print(value)
                                    data[key] = value
                    data_received = None
            except BlockingIOError:
                pass
            

            # if data and data['tasks'] is not None:
            #     tasks = data['tasks']
                # print('Received tasks from server:', tasks)

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Victim handling
                if gui.button_accept.handle_event(event):
                    gui.image = None
                    # data['idx_image'] = None
                    if victim_buffer:
                        victim_buffer.pop(0)
                    response_changed = True
                    response['victim'] = 'accept'
                elif gui.button_reject.handle_event(event):
                    gui.image = None
                    # data['idx_image'] = None
                    if victim_buffer:
                        victim_buffer.pop(0)
                    response_changed = True
                    response['victim'] = 'reject'
                elif gui.button_handover.handle_event(event):
                    gui.image = None
                    # data['idx_image'] = None
                    if victim_buffer:
                        victim_buffer.pop(0)
                    response_changed = True
                    response['victim'] = 'handover'
                else:
                    pass

                ########## Humane handling ##############
                ########## Human handling ends ##########


                # Task handling
                for task in gui.task_list:
                    if task.handle_event(event):
                        response_changed = True

                # # Remove rejected tasks
                # gui.task_list = [ta for ta in gui.task_list if not ta.reject]

                if response_changed:
                    response['tasks'] = [task.to_dict() for task in gui.task_list]

                # Weather handling
                if gui.button_wind_change.handle_event(event):
                    response_changed = True
                    response['weather_decision'] = 'change'
                elif gui.button_wind_maintain.handle_event(event):
                    response_changed = True
                    response['weather_decision'] = 'maintain'
                elif gui.button_wind_handover.handle_event(event):
                    response_changed = True
                    response['weather_decision'] = 'handover'
                else:
                    pass
                    
                # Response handling
                gui.response_input.handle_event(event)
                if gui.response_input.finish:
                    # response_changed = True
                    # response['vic_response'] = gui.response_input.text_send
                    gui.response_input.finish = False  # Reset finish flag
                    gui.response_input.text = ""
                    # print(vic_msg_buffer)
                    if vic_msg_buffer:
                        vic_msg_buffer.pop(0)
                        # print(vic_msg_buffer)

            # Render the GUI and get the response
            gui.render()
            idx_image = None  # Reset image after rendering

            # Send response back to the server if it has changed
            if response_changed:
                print(response['tasks'])
                msg = json.dumps(response) + '\n'
                s.sendall(msg.encode('utf-8'))
    finally:
        s.close()
        pygame.quit()
