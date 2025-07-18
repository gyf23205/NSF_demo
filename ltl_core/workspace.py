import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from .agent import Agent
from .specification import get_ap_prefix, AP_TYPE_PREFIX_MAP


class Workspace:
    def __init__(self, size=(30, 30), target_mask=None,
                 num_drones=2, num_gvs=2, num_humans=2, seed=119, margin=3):
        self.size = size
        self.grid = np.zeros(size, dtype=int)
        self.rng = random.Random(seed)

        self.target_mask = target_mask or []
        self.margin = margin

        self.base_area = [(x, y) for x in range(1, 4) for y in range(2, 5)]
        self.hospital_area = [(x, y) for x in range(size[0] - 4, size[0] - 1)
                              for y in range(2, 5)]

        self._place_targets()
        self.agent_positions = set()

        self.agents = {
            "humans": self._place_humans_deterministically(num_humans),
        }
        self.agent_positions.update([tuple(agent.pos) for agent in self.agents["humans"]])

        self.agents["drones"] = self._place_agents_in_base(num_drones, role="drones")
        self.agents["gvs"] = self._place_agents_in_base(num_gvs, role="gvs")

        self.dropoff_locations = self._assign_dropoff_locations()

        self._active_aps = set()    # Holds currently true atomic propositions
        self.true_aps = set()

    @staticmethod
    def _place_humans_deterministically(num):
        fixed = []
        candidates = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
        for i in range(num):
            label = f"H{i}"
            fixed.append(Agent(pos=np.array(candidates[i % len(candidates)], dtype=float), role="human", label=label))
        return fixed

    def _place_agents_in_base(self, num, role):
        available = [pos for pos in self.base_area if pos not in self.agent_positions]
        self.rng.shuffle(available)
        agents = []
        for i in range(num):
            if not available:
                raise ValueError("Not enough space in base area to place all agents without overlap.")
            label = f"{role[0].upper()}{i}"
            pos = available.pop()
            np_pos = np.array(pos, dtype=float)
            agent = Agent(pos=np_pos, role=role, label=label)
            agents.append(agent)
            self.agent_positions.add(tuple(np_pos))

        return agents

    def _place_targets(self):
        num_targets = len(self.target_mask)
        forbidden = set(self.base_area + self.hospital_area)
        possible_positions = self._valid_positions(self.margin, exclude=forbidden)
        self.rng.shuffle(possible_positions)

        self.target_locations = []
        for i in range(num_targets):
            if self.target_mask[i] and possible_positions:
                while possible_positions:
                    candidate = possible_positions.pop()
                    if all(self._distance(candidate, t) >= self.margin for t in self.target_locations):
                        self.target_locations.append(candidate)
                        break

    def _valid_positions(self, margin=0, exclude=None):
        rows, cols = self.size
        exclude = set(exclude or [])
        return [(x, y) for x in range(margin, rows - margin)
                for y in range(margin, cols - margin)
                if (x, y) not in exclude]

    def _assign_dropoff_locations(self):
        """Assign each target a unique hospital cell for dropoff."""
        assigned = {}
        for i, cell in enumerate(self.hospital_area):
            assigned[i] = cell  # tid → (x, y)
            if i + 1 >= len(self.target_mask):
                break
        return assigned

    @staticmethod
    def _distance(pos1, pos2):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        return np.linalg.norm(pos1 - pos2)

    def plot(self):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.size[1])
        ax.set_ylim(0, self.size[0])
        ax.set_aspect('equal')
        ax.grid(True)

        # Draw base area
        for (x, y) in self.base_area:
            ax.add_patch(patches.Rectangle((y, x), 1, 1,
                                           linewidth=1, edgecolor='blue',
                                           facecolor='lightblue', alpha=0.3,
                                           label="Base" if (x, y) == self.base_area[0] else "_nolegend_"))

        # Draw hospital area
        for (x, y) in self.hospital_area:
            ax.add_patch(patches.Rectangle((y, x), 1, 1,
                                           linewidth=1, edgecolor='red',
                                           facecolor='mistyrose', alpha=0.3,
                                           label="Hospital" if (x, y) == self.hospital_area[0] else "_nolegend_"))

        # Draw target locations
        for i, (x, y) in enumerate(self.target_locations):
            ax.plot(y + 0.5, x + 0.5, 'ro', markersize=10)
            ax.text(y + 0.5, x + 0.5, f"T{i}", color='black',
                    fontsize=12, ha='center', va='center')

        # Draw agents
        for role, agents in self.agents.items():
            color = {'drones': 'b', 'gvs': 'g', 'humans': 'orange'}[role]
            for agent in agents:
                x, y = int(agent.pos[0]), int(agent.pos[1])
                label = agent.label

                # Get progress of current symbolic task, if any
                task = agent.current_symbolic_task
                p_val = agent.get_progress(task) if task else 0.0

                # Draw agent base
                ax.plot(y + 0.5, x + 0.5, 'o', color=color, markersize=10)
                ax.text(y + 0.5, x + 0.5, label, fontsize=12, ha='center', va='center')

                # Draw progress ring
                if p_val > 0:
                    ax.plot(y + 0.5, x + 0.5, 'o', color='lime', markersize=15, alpha=p_val * 0.6)
                    ax.text(y + 0.5, x + 0.8, f"{p_val:.1f}", fontsize=8, color='green', ha='center')

        ax.set_xticks(range(self.size[1]))
        ax.set_yticks(range(self.size[0]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.title("Workspace")
        ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()
        plt.show()

    def set_ap_true(self, ap: str):
        """Mark an AP as true for toy testing."""
        self._active_aps.add(ap)

    def get_true_aps(self):
        return self.true_aps

    def reset(self):
        self._active_aps = set()
        for agent in self.get_all_agents():
            agent.reset()
        return self

    def get_all_agents(self):
        return self.agents["drones"] + self.agents["gvs"] + self.agents["humans"]

    def update_true_aps(self):
        true_aps = set()

        # --- Physical APs: GVs and Drones ---
        for agent in self.agents["drones"] + self.agents["gvs"]:
            if not agent.has_arrived():
                continue

            # Identify goal match for pickup/dropoff/nav
            if agent.role == "drones":
                for idx, loc in enumerate(self.target_locations):
                    if np.allclose(agent.goal, loc):
                        ap = f"p_nav_{idx}_1_1_0"
                        true_aps.add(ap)

            elif agent.role == "gvs":
                # Check for pickup (at target)
                for idx, loc in enumerate(self.target_locations):
                    if np.allclose(agent.goal, loc):
                        ap = f"p_pickup_{idx}_2_1_0"
                        true_aps.add(ap)

                # Check for dropoff (at hospital)
                for idx, loc in self.dropoff_locations.items():
                    if np.allclose(agent.goal, loc):
                        ap = f"p_dropoff_{idx}_2_1_0"
                        true_aps.add(ap)

        # --- Symbolic APs ---
        for agent in self.get_all_agents():
            task = agent.current_symbolic_task
            if task is None:
                continue
            prefix = get_ap_prefix(task)
            ap_type = AP_TYPE_PREFIX_MAP.get(prefix)
            if ap_type == "symbolic" and agent.has_completed(task):
                true_aps.add(task)

        self.true_aps = true_aps

    def step_dynamics(self, dt=1.0):
        for agent in self.get_all_agents():
            agent.move_toward_goal(dt)
            agent.step_symbolic(dt)
        self.update_true_aps()

    def meter_to_pixel(self, pos, screen_size=(1200, 900)):
        """
        Convert a (x, y) position in meters to pixel coordinates.
        Assumes (0,0) is bottom-left and y increases upward.
        """
        x_m, y_m = pos
        ws_w, ws_h = self.size
        scr_w, scr_h = screen_size

        # Scale factors
        scale_x = scr_w / ws_w
        scale_y = scr_h / ws_h

        px = int(x_m * scale_x)
        py = scr_h - int(y_m * scale_y)  # flip Y for Pygame

        return px, py
    
    def meters_to_pixels(self, positions, screen_size=(1200, 900)):
        return [self.meter_to_pixel(pos, screen_size) for pos in positions]
    
    @staticmethod
    def grid_to_pixel(grid_pos, grid_size=(50, 40), screen_size=(900, 720)):
        """
        Convert from (x, y) in grid units to (px, py) in pixel coordinates.
        Assumes (0, 0) is bottom-left of screen and Y increases upward in grid.
        """
        x_grid, y_grid = grid_pos
        grid_w, grid_h = grid_size
        screen_w, screen_h = screen_size

        scale_x = screen_w / grid_w
        scale_y = screen_h / grid_h

        px = int(x_grid * scale_x)
        py = screen_h - int(y_grid * scale_y)  # flip Y for Pygame

        return px, py
    
    @staticmethod
    def grid_to_game_mgr(pos, grid_size=(50, 40)):
        """
        Convert from grid-space meters (e.g. [0, 50] * [0, 40]) 
        to GameMgr-style world meters (e.g. [-2, 2] * [-1.5, 1.5]).
        """
        x, y = pos
        x_scaled = (x / grid_size[0]) * 4.0 - 2.0      # maps 0→50 → -2→2
        y_scaled = (y / grid_size[1]) * 3.0 - 1.5      # maps 0→40 → -1.5→1.5
        return np.array([x_scaled, y_scaled])