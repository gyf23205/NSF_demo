import numpy as np
from ltl_core.dag_builder import build_dag
from ltl_core.automaton_generator import compile_automata
from ltl_core.specification import get_ap_prefix, AP_TYPE_PREFIX_MAP

class Simulation:
    def __init__(self, spec, workspace, allocator, labeler):
        self.spec = spec
        self.workspace = workspace
        self.allocator = allocator
        self.binding_manager = spec.binding_manager
        self.labeler = labeler

        # Build the DAG and compile automata
        self.dag = spec.dag
        self.automata = spec.automata

        # Episode trace for logging
        self.episode_trace = []

        # Wait for human verification responses
        self.verify_response_pending = set()

        # Check logical progress
        self.prev_completed = set()
        self.prev_actions = {}

    @staticmethod
    def parse_ap_target_index(ap: str) -> int:
        return int(ap.split("_")[2])

    def get_agent_by_label(self, label: str):
        for agent in self.workspace.get_all_agents():
            if agent.label == label:
                return agent
        return None
    
    def step(self, dt, verbose=False):
        # Advance agent dynamics
        self.workspace.step_dynamics(dt=dt)

        # Label APs and update DFA
        state = self.workspace
        aps = self.labeler.extract_APs(state)
        self.labeler.advance(aps)

        unlocked = self.labeler.get_unlocked_APs()
        completed = self.labeler.get_completed()
        current_aps = self.labeler.current_aps

        if completed != self.prev_completed:
            self.prev_actions = self.allocator.choose(unlocked, completed, current_aps)
            self.prev_completed = completed
        actions = self.prev_actions

        if verbose:
            print(f"[GUI-STEP] Unlocked: {sorted(unlocked)}")
            print(f"[GUI-STEP] Assigned: {[f'{a.label}→{ap}' for a, ap in actions.items()]}")
            print(f"[GUI-STEP] Completed: {sorted(completed)}")

        # Apply assignments (symbolic and physical)
        for agent, ap in actions.items():
            prefix = get_ap_prefix(ap)
            ap_type = AP_TYPE_PREFIX_MAP.get(prefix)

            # Reset goal
            agent.goal = None

            if ap_type == "physical":
                idx = self.parse_ap_target_index(ap)
                if prefix == "p_dropoff":
                    agent.goal = np.array(self.workspace.dropoff_locations[idx], dtype=float)
                else:
                    agent.goal = np.array(self.workspace.target_locations[idx], dtype=float)
            
            elif ap_type == "symbolic":
                # Block symbolic APs if they're pending verification
                if hasattr(self, "verify_response_pending"):
                    if ap in self.verify_response_pending:
                        continue  # Skip this AP until human responds
                if agent.current_symbolic_task is None:
                    agent.start_symbolic_task(ap)
                    agent.set_symbolic_task_speed(ap, speed=0.1)

        # Idle return-to-base for unassigned drones/GVs
        for agent in self.workspace.get_all_agents():
            if agent.role not in ["drones", "gvs"]:
                continue

            is_idle = (
                (agent.goal is None or agent.has_arrived())
                and tuple(agent.pos.astype(int)) not in self.workspace.base_area
            )

            if is_idle:
                occupied = {
                    tuple(a.goal.astype(int)) for a in self.workspace.get_all_agents()
                    if a.goal is not None
                }
                taken_now = {
                    tuple(a.pos.astype(int)) for a in self.workspace.get_all_agents()
                }
                unavailable = occupied | taken_now
                available_bases = [b for b in self.workspace.base_area if b not in unavailable]

                if available_bases:
                    agent.goal = np.array(available_bases[0], dtype=float)

        # Mark symbolic task complete
        for agent in self.workspace.get_all_agents():
            task = agent.current_symbolic_task
            if task and task in completed:
                agent.current_symbolic_task = None

        # Optional logging
        self.episode_trace.append({
            "aps": sorted(current_aps),
            "completed": completed
        })

        # print("[Labeling] APs = ", sorted(aps))
        # print(sorted(list(completed)))

        return {
            "unlocked": unlocked,
            "completed": completed,
            "assignments": actions,
            "label": aps
        } 
