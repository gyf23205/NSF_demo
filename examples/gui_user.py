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

# For workload estimation
csv_path = 'dummy_log\\aggregated_output.csv'
# csv_path = 'C:/Users/JW Choi/Desktop/NSF_2025_demo/dataset/aggregated_output.csv'
from realtime_heart_plot import RealtimeHeartPlot
from workload_speedometer import WorkloadSpeedometer


class Task:
    def __init__(self, surface, task_id, target_loc, task_pos, priority=0):
        self.task_id = task_id
        self.surface = surface
        self.x0, self.y0 = task_pos
        self.target_pos = target_loc
        self.priority = priority
        self.assigned_drone = None
        # self.assigned_gv = None
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
        # self.assigned_gv_text = Font(FONT, FONT_SIZE, (self.x0 + 4 * (self.grid_width + self.spacing), self.y0))
        # self.rejection_button = Button((self.x0 + 5 * (self.grid_width + self.spacing), self.y0, self.grid_width, self.grid_height), RED, "Reject", text_color=WHITE)
        # self.assigned_gv_input = TextInput((self.x0 + 3 * self.grid_width, self.y0, self.grid_width, self.grid_height), color=WHITE, maximum=n_gvs)
        
        self.task_id_text.update('    ' + f'{self.task_id}')
        self.target_pos_text.update(f'{self.target_pos}')
        self.priority_input.text = str(self.priority)
        self.assigned_drone_text.update(str(self.assigned_drone))
        # self.assigned_gv_text.update(str(self.assigned_gv))

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
        # self.assigned_gv_text.pos = (self.x0 + 4 * (self.grid_width + self.spacing), self.y0)
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

        # self.assigned_gv_text.clear()
        # self.assigned_gv_text.update(str(self.assigned_gv))
        # for text in self.assigned_gv_text.texts:
            # self.surface.blit(text[0], text[1])

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
        self.screen_height = 850
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill(WHITE)

        # Victim block
        self.image_width = 420
        self.image_height = 420
        self.image_rect = pygame.Rect(
            110,
            200,
            self.image_width,
            self.image_height
        )
        # Center the buttons below the image
        button_width = 120
        button_height = 50
        spacing = 100
        buttons_y = self.image_rect.bottom + 10
        # total_buttons_width = button_width * 2 + spacing
        # buttons_x = self.image_rect.x + (0.5*self.image_width - button_width - 0.5*spacing) - 120
        buttons_x = 150
        accept_rect = pygame.Rect(buttons_x, buttons_y, button_width, button_height)
        reject_rect = pygame.Rect(buttons_x + button_width + spacing, buttons_y, button_width, button_height)
        # handover_rect = pygame.Rect(buttons_x + 2 * (button_width + spacing), buttons_y, button_width, button_height)
        self.button_accept =  Button(accept_rect, BLUE, 'Survivor', text_color=WHITE)
        self.button_reject = Button(reject_rect, RED, 'Negative', text_color=WHITE)
        # self.button_handover = Button(handover_rect, GREEN, 'Hand over', text_color=WHITE)
        self.image = None
        self.button_accept.draw(self.screen)
        self.button_reject.draw(self.screen)
        # self.button_handover.draw(self.screen)

        # Workload block
        # self.workload_text = Font(FONT, FONT_SIZE, (50, 200))
        # self.workload = 'low'
        # self.workload_text.update('Workload: '+ self.workload)
        # self.screen.blit(self.workload_text.texts[0][0], self.workload_text.texts[0][1])
        
        # weather block (-> temporarily used as Survivor Triage)
        weather_x = 50
        weather_y = 710
        self.weather_text = Font(FONT, FONT_SIZE, (weather_x, weather_y))
        self.weather = 'sunny'
        self.weather_text.update('Weather: '+ self.weather)
        self.button_wind_change = Button((weather_x, weather_y + FONT_SIZE * line_height + 30, button_width, button_height), RED, "Emergency", text_color=BLACK)
        self.button_wind_maintain = Button((weather_x + button_width + spacing, weather_y + FONT_SIZE * line_height + 30, button_width, button_height), YELLOW, "Injury", text_color=BLACK)
        self.button_wind_handover = Button((weather_x + 2 * (button_width + spacing), weather_y + FONT_SIZE * line_height + 30, button_width, button_height), GREEN, "Minor", text_color=BLACK)
        self.button_wind_change.draw(self.screen)
        self.button_wind_maintain.draw(self.screen)
        self.button_wind_handover.draw(self.screen)
        # self.wind_speed_received = None

        # Survivor Triage block (top-right)
        self.surv_header_pos = (self.weather_text.pos[0], self.weather_text.pos[1])
        self.surv_msg_pos = (self.surv_header_pos[0], self.surv_header_pos[1] + FONT_SIZE * line_height)

        # Keep dedicated fonts and rects to avoid overlapping draws
        self.surv_header_font = Font(FONT, FONT_SIZE, self.surv_header_pos)
        self.surv_msg_font = Font(FONT, FONT_SIZE, self.surv_msg_pos)

        # Compute a safe clear area for the survivor header + one line of text
        self.surv_clear_rect = pygame.Rect(
            self.surv_header_pos[0],
            self.surv_header_pos[1],
            3 * (grid_width + spacing) + 400,   # wide enough to cover header + one line msg
            2 * FONT_SIZE * line_height + 10
)
        # Task block
        self.task_x = 720
        self.task_y = 270
        # self.received_new_tasks = False
        self.task_text = Font(FONT, FONT_SIZE, (self.task_x, self.task_y))
        self.task_text.update('                                                Task Monitor')
        self.task_text.update('Task ID                 Target pos              Priority                Assigned Drone')
        for text in self.task_text.texts:
            self.screen.blit(text[0], text[1])
        self.task_list_x = self.task_x
        self.task_list_y = self.task_y + len(self.task_text.texts) * line_height * FONT_SIZE
        self.task_list = []
        self.n_previous_tasks = len(self.task_list)
        self.tasks_received = None

        # --- Fire→Priority block (text + input) just above Task Monitor ---
        # Position it a bit above the task header area
        self.fire_x = self.task_x
        self.fire_y = self.task_y - 3.5 * FONT_SIZE * line_height

        self.fire_title = Font(FONT, FONT_SIZE, (self.fire_x, self.fire_y))
        self.fire_body  = Font(FONT, FONT_SIZE, (self.fire_x, self.fire_y + FONT_SIZE * line_height))
        # Input for numeric priority
        self.fire_input = TextInputResponse((self.fire_x, self.fire_y + 2 * FONT_SIZE * line_height,
                                            220, FONT_SIZE * line_height), color=WHITE, maximum=3)

        # Region to clear each frame
        self.fire_rect = pygame.Rect(self.fire_x, self.fire_y,
                             6 * (grid_width + spacing), 3 * FONT_SIZE * line_height + 10)

        # Response block
        response_x = 720
        response_y = 30
        self.response_title = Font(FONT, FONT_SIZE, (response_x, response_y))
        self.response_title.update('Air Traffic Management')
        # self.screen.blit(self.response_title.texts[0][0], self.response_title.texts[0][1])
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
        state_dict = torch.load('last_gauge.pt', map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        ecg = last_row[:130]
        gaze_au_matrix = np.array(last_row[130:]).reshape(10, 30)

        t1 = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0) # raw ECG
        t2 = torch.tensor(gaze_au_matrix, dtype=torch.float32)  # [10, 30]

        if torch.isnan(t2).any() or torch.isinf(t2).any():
            print("NaN or Inf detected in t2 (gaze input)")

        with torch.no_grad():
            out = model(t1, t2)
            pred_label = torch.argmax(out).item()
            # print(out, pred_label)

        # 3. update workload  
        # if pred_label > 0.5:
        #     workload_text = 'high'
        # else:
        #     workload_text = 'low'

        # self.workload_text.clear()
        # self.workload_text.update('Workload: ' + workload_text)  
        # area = self.workload_text.rect.copy()
        # area.width += 100  # Adjust width to fit the screen
        # pygame.draw.rect(self.screen, WHITE, area)
        # self.screen.blit(self.workload_text.texts[0][0], self.workload_text.texts[0][1])          

        # Render meter graphics
        heart_plot.update(ecg); heart_plot.render(self.screen)                                     # ONE LINE for HR
        workload_meter.update(pred_label)
        workload_meter.render(self.screen, (100, 80)) # ONE LINE for WL   
        ###################### Update workload text ends #####################


        ###################### Victim block ######################
        # if data and data['idx_image'] is not None:
        #     # print(data['idx_image'])
        #     victim_buffer.append(data['idx_image'])
        #     data['idx_image'] = None
        if victim_buffer:
            image_id = victim_buffer[0]["image_id"] if isinstance(victim_buffer[0], dict) else victim_buffer[0]
            image_path = f"examples/images/victim{image_id}.jpg"
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
        
        ###################### Fire→Priority block (above Task Monitor) ######################
        pygame.draw.rect(self.screen, WHITE, self.fire_rect)

        # Title
        self.fire_title.clear()
        self.fire_title.update('Priority Request')
        self.screen.blit(self.fire_title.texts[0][0], self.fire_title.texts[0][1])

        # Body & input
        self.fire_body.clear()
        if fire_active and fire_text:
            self.fire_body.update(fire_text)  # e.g., "Region 3 is in danger. Set its priority to HIGH (2)."
        else:
            self.fire_body.update('No active fire-related priority request.')
        self.screen.blit(self.fire_body.texts[0][0], self.fire_body.texts[0][1])

        # Input box (always drawn, but only meaningful if active)
        self.fire_input.draw(self.screen)
        ###################### Fire→Priority block ends ######################

        ###################### Task block ######################
        # print('data before receiving tasks: ', data)
        if self.tasks_received is not None:
            self.task_list = []
            for i, task in enumerate(self.tasks_received):
                task_pos = (self.task_list_x, self.task_list_y + i * FONT_SIZE * line_height)
                # [task_id, [x,y], priority, assigned_drone, assigned_gv]
                task_id, target_loc, priority, assigned_drone, assigned_gv = task
                new_task = Task(self.screen, task_id, target_loc, task_pos, priority)
                new_task.assigned_drone = assigned_drone  # may be None
                # new_task.assigned_gv = assigned_gv        # may be None
                self.task_list.append(new_task)
            self.tasks_received = None

        # Clear the task area before drawing
        task_table_width = 6 * (grid_width + spacing)
        task_table_height =  self.screen_height - self.task_y
        # print(f'Number of previous tasks: {self.n_previous_tasks}')
        pygame.draw.rect(self.screen, WHITE, (self.task_x, self.task_y + 2 * line_height * FONT_SIZE, task_table_width, task_table_height))
        
        if self.task_list:
            # assert False
            if self.n_previous_tasks != len(self.task_list):
                task_list_temp = []
                for i, task in enumerate(self.task_list):
                    task_pos = (self.task_list_x, self.task_list_y + i * FONT_SIZE * line_height)
                    task_id = task.task_id
                    target_loc = task.target_pos
                    priority = task.priority
                    assigned_drone = task.assigned_drone
                    # assigned_gv = task.assigned_gv
                    new_task = Task(self.screen, task_id, target_loc, task_pos, priority)
                    new_task.assigned_drone = assigned_drone
                    # new_task.assigned_gv = assigned_gv
                    task_list_temp.append(new_task)
                self.task_list = task_list_temp
                self.n_previous_tasks = len(self.task_list)

            # Now draw the updated task list
            for task in self.task_list:
                task.draw()
        ########################## Task block ends ######################

        ###################### Survivor Triage (top-right) ######################
        # Clear previously drawn header + message area to prevent overlap
        pygame.draw.rect(self.screen, WHITE, self.surv_clear_rect)

        # Header
        self.surv_header_font.clear()
        self.surv_header_font.update('Survivor Triage')
        self.screen.blit(self.surv_header_font.texts[0][0], self.surv_header_font.texts[0][1])

        # Message line below header
        self.surv_msg_font.clear()
        self.surv_msg_font.update(surv_text if surv_active and surv_text else "No active survivor message.")
        self.screen.blit(self.surv_msg_font.texts[0][0], self.surv_msg_font.texts[0][1])

        # Buttons: keep using existing buttons; just redraw them each frame
        self.button_wind_change.text   = surv_choices[0]  # Emergency
        self.button_wind_maintain.text = surv_choices[1]  # Serious
        self.button_wind_handover.text = surv_choices[2]  # Minor
        self.button_wind_change.draw(self.screen)
        self.button_wind_maintain.draw(self.screen)
        self.button_wind_handover.draw(self.screen)
        ###################### Survivor Triage ends ######################

        ####################### Response block ##########################
        # Use the current title position as the block's origin
        response_x, response_y = self.response_title.pos
        line_h = int(FONT_SIZE * line_height)

        # Clear a fixed wide area (from response_x to screen right)
        response_region_width  = self.screen_width - response_x - 10
        response_region_height = 3 * line_h + 30   # title + one line + padding + input
        pygame.draw.rect(self.screen, WHITE, (response_x, response_y, response_region_width, response_region_height))

        # Title
        self.response_title.clear()
        self.response_title.update('Air Traffic Management')
        self.screen.blit(self.response_title.texts[0][0], self.response_title.texts[0][1])

        # Body
        self.response_text.clear()
        msg = atm_text if atm_active and atm_text else 'No active ATM message.'
        self.response_text.update(msg)
        self.screen.blit(self.response_text.texts[0][0], self.response_text.texts[0][1])

        # Input just below
        self.response_input.rect.topleft = (response_x, response_y + 2 * line_h + 10)
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

    # Meter Graphics initailization (ECG graphic)
    heart_plot = RealtimeHeartPlot(position=(250, 20), hr_interval_seconds= 10)
    workload_meter = WorkloadSpeedometer(850, 500)

    # response['victim']: 'reject' or 'accept'
    # response['weather_decision']: 'change' or 'maintain'
    # response['tasks']: list of Task objects
    response = {'victim': None, 'weather_decision': None, 'tasks': None} # Response to be sent back to the server
    running = True

    # Survivor image
    victim_buffer = []  # Buffer to store victims
    vic_msg_buffer = []  # Buffer to store messages from victims

    # ATM message
    atm_active = False
    atm_prompt_id = None
    atm_token = None
    atm_text = ""   # what we show in the left-bottom message line

    # Survivor triage (top-right)
    surv_active = False
    surv_prompt_id = None
    surv_text = ""
    surv_choices = ["Emergency", "Serious", "Minor"]

    # Fire→Priority (right-bottom, above Task Monitor)
    fire_active = False
    fire_prompt_id = None
    fire_task_id = None
    fire_required = None
    fire_text = ""

    # data = {'idx_image': None, 'tasks': None, 'wind_speed': None, 'vic_msg': None}  # Initialize data
    try:
        recv_buffer = ''
        while running:
            response_changed = False
            # Receive weather, task, victim from server
            try:
                data_received = s.recv(4096).decode()
                # print('Data received !!!!!!!!!!!!!!!!!!!!!!!!!!')
                if data_received:
                    recv_buffer += data_received
                    while '\n' in recv_buffer:
                        line, recv_buffer = recv_buffer.split('\n', 1)
                        if line.strip():
                            data_temp = json.loads(line)
                            # print('Received data from server:', repr(data_temp))
                            for key, value in data_temp.items():
                                if value is not None:
                                    if key == 'idx_image':
                                        # value is {"image_id": int, "tid": "2"} from server
                                        victim_buffer.append(value)
                                    if key == 'tasks':
                                        gui.tasks_received = value
                                    # if key == 'wind_speed':
                                        # gui.wind_speed_received = value
                                    # if key == 'vic_msg':
                                    #     vic_msg_buffer.append(value)
                                    if key == 'atm_prompt':
                                        # value: {"id": ..., "text": "...", "token": "...", "coord": [x, y]}
                                        atm_active = True
                                        atm_prompt_id = value.get("id")
                                        atm_token = value.get("token")
                                        atm_text = value.get("text") or ""
                                    if key == 'atm_clear':
                                        # value: {"id": ...}
                                        atm_active = False
                                        atm_prompt_id = None
                                        atm_token = None
                                        atm_text = ""  # will fall back to existing buffer or placeholder
                                    if key == 'surv_prompt':
                                        # {"id","text","choices":[...]}
                                        surv_active = True
                                        surv_prompt_id = value.get("id")
                                        surv_text = value.get("text") or ""
                                        surv_choices = value.get("choices") or ["Emergency","Serious","Minor"]
                                    if key == 'surv_clear':
                                        # {"id":..., "ok": True|False}
                                        if value.get("id") == surv_prompt_id:
                                            surv_active = False
                                            surv_prompt_id = None
                                            surv_text = ""
                                    if key == 'fire_prompt':
                                        # {"id","task_id","text","required"}
                                        fire_active = True
                                        fire_prompt_id = value.get("id")
                                        fire_task_id = value.get("task_id")
                                        fire_required = value.get("required")
                                        fire_text = value.get("text") or ""
                                    if key == 'fire_clear':
                                        # {"id":..., "ok": True|False, "reason": "..."}
                                        if value.get("id") == fire_prompt_id:
                                            # clear regardless of ok to match server contract
                                            fire_active = False
                                            fire_prompt_id = None
                                            fire_task_id = None
                                            fire_required = None
                                            fire_text = ""
                    data_received = None
            except BlockingIOError:
                pass

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Victim handling
                if gui.button_accept.handle_event(event):
                    gui.image = None
                    clicked_item = victim_buffer.pop(0) if victim_buffer else None
                    response_changed = True
                    response['victim'] = 'accept'
                    response['verify_tid'] = (
                        str(clicked_item['tid']) if isinstance(clicked_item, dict) and 'tid' in clicked_item else None
                    )

                elif gui.button_reject.handle_event(event):
                    gui.image = None
                    clicked_item = victim_buffer.pop(0) if victim_buffer else None
                    response_changed = True
                    response['victim'] = 'reject'
                    response['verify_tid'] = (
                        str(clicked_item['tid']) if isinstance(clicked_item, dict) and 'tid' in clicked_item else None
                    )

                # elif gui.button_handover.handle_event(event):
                #     gui.image = None
                #     clicked_item = victim_buffer.pop(0) if victim_buffer else None
                #     response_changed = True
                #     response['victim'] = 'handover'
                #     response['verify_tid'] = (
                #         str(clicked_item['tid']) if isinstance(clicked_item, dict) and 'tid' in clicked_item else None
                #     )

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

                # Top-right triage buttons
                if gui.button_wind_change.handle_event(event):
                    if surv_active and surv_prompt_id is not None:
                        s.sendall((json.dumps({"surv_reply": {"id": surv_prompt_id, "choice": surv_choices[0]}}) + '\n').encode('utf-8'))

                elif gui.button_wind_maintain.handle_event(event):
                    if surv_active and surv_prompt_id is not None:
                        s.sendall((json.dumps({"surv_reply": {"id": surv_prompt_id, "choice": surv_choices[1]}}) + '\n').encode('utf-8'))

                elif gui.button_wind_handover.handle_event(event):
                    if surv_active and surv_prompt_id is not None:
                        s.sendall((json.dumps({"surv_reply": {"id": surv_prompt_id, "choice": surv_choices[2]}}) + '\n').encode('utf-8'))
                    
                # ATM response handling
                gui.response_input.handle_event(event)
                if gui.response_input.finish:
                    user_text = getattr(gui.response_input, 'text_send', gui.response_input.text)
                    gui.response_input.finish = False
                    gui.response_input.text = ""

                    if atm_active and atm_prompt_id is not None:
                        s.sendall((json.dumps({"atm_reply": {"id": atm_prompt_id, "typed": user_text}}) + '\n').encode('utf-8'))
                
                # Fire→Priority input handling
                gui.fire_input.handle_event(event)
                if gui.fire_input.finish:
                    user_text = getattr(gui.fire_input, 'text_send', gui.fire_input.text).strip()
                    gui.fire_input.finish = False
                    gui.fire_input.text = ""
                    if fire_active and fire_prompt_id is not None and fire_task_id is not None:
                        try:
                            priority_val = int(user_text)
                        except ValueError:
                            priority_val = user_text
                        s.sendall((json.dumps({
                            "priority_reply": {
                                "id": fire_prompt_id,
                                "task_id": int(fire_task_id),
                                "priority": priority_val
                            }
                        }) + '\n').encode('utf-8'))

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
