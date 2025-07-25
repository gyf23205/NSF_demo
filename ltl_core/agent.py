import numpy as np


class Agent:
    def __init__(self, pos, role, label=None, speed=10.0):
        self.pos = np.array(pos, dtype=float)   # Continuous 2D position
        self.role = role                        # "drone", "gv", or "human"
        self.label = label                      # Optional string label like "D0", "H1"
        self.traj = [tuple(self.pos)]           # Continuous position trace

        self.goal = None                        # Target position (x,y) or None
        self.speed = speed                      # Units per timestep

        self.current_symbolic_task = None       # Currently executing symbolic task (str)
        self.symbolic_task_speed = {}           # task_name → speed (0.1, 1.0, etc.)
        self.symbolic_progress = {}             # task_name → progress [0.0 ~ 1.0]

        self.scan_center = None                 # Center of scan area (x,y)
        self.scan_angle = 0.0                   # Scan angle in degrees
        self.scan_time = 0.0                    # Time spent in scan area

    def move_toward_goal(self, dt=1.0):
        """Move the agent toward its current goal, if any."""
        if self.goal is None:
            return
        direction = self.goal - self.pos
        dist = np.linalg.norm(direction)
        if dist < 1e-5:
            return
        step = self.speed * dt
        if step >= dist:
            self.pos = self.goal.copy()
        else:
            self.pos += direction / dist * step

    def set_symbolic_task_speed(self, task_name: str, speed: float):
        """Set the speed for symbolic task execution."""
        self.symbolic_task_speed[task_name] = speed

    def start_symbolic_task(self, task_name: str):
        """Start a symbolic task. Only one symbolic task can run at a time."""
        if self.current_symbolic_task is not None:
            raise RuntimeError(f"Agent {self.label} is already doing symbolic task {self.current_symbolic_task}")
        self.current_symbolic_task = task_name
        if task_name not in self.symbolic_progress:
            self.symbolic_progress[task_name] = 0.0

    def stop_symbolic_task(self):
        """Stop the currently executing symbolic task."""
        self.current_symbolic_task = None

    def step_symbolic(self, dt=1.0):
        """Progress the currently executing symbolic task by dt * speed."""
        task = self.current_symbolic_task
        if task is None:
            return
        speed = self.symbolic_task_speed.get(task, 1.0)
        self.symbolic_progress[task] = min(1.0, self.symbolic_progress[task] + speed * dt)

    def get_progress(self, task_name: str) -> float:
        """Return progress [0.0 ~ 1.0] for the specified symbolic task."""
        return self.symbolic_progress.get(task_name, 0.0)

    def has_completed(self, task_name: str) -> bool:
        """Check if symbolic task has been completed (progress == 1.0)."""
        return self.get_progress(task_name) >= 1.0

    def has_arrived(self, tol: float = 1e-5) -> bool:
        """Check if agent is within tol distance of a goal position."""
        if self.goal is None:
            return False
        dist = np.linalg.norm(self.pos - self.goal)
        # print(f"[ARRIVAL CHECK] {self.label} dist={dist:.3f} tol={tol}")
        return dist < tol

    def reset(self):
        """Reset agent's position trace and symbolic state."""
        self.traj = [tuple(self.pos)]
        self.goal = None
        self.current_symbolic_task = None
        self.symbolic_progress.clear()

    def __repr__(self):
        return f"Agent(label={self.label}, role={self.role}, pos={self.pos}, progress={self.symbolic_progress:.2f})"
