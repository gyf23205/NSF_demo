from .agent import Agent
from collections import defaultdict
from typing import Dict, List, Optional, Set
import networkx as nx


class BindingManager:
    """
    Tracks binding constraints between level-4 functions and agent IDs/types.
    Ensures consistent assignment of agents to logically grouped tasks.
    """

    def __init__(self, enforce_default=True, verbose=False):
        self.enforce_default = enforce_default
        self.verbose = verbose

        # task_name -> group_key
        self.task_to_group: Dict[str, str] = {}

        # group_key -> list of task_names
        self.group_to_tasks: Dict[str, Set[str]] = defaultdict(set)

        # group_key -> {agent_type: agent_id}
        self.bindings: Dict[str, Dict[str, str]] = defaultdict(dict)

        # group_key -> allowed agent types (e.g., {"drone"}, {"gv"}, {"drone", "human"})
        self.group_types: Dict[str, Set[str]] = {}

        # task_name with explicitly disabled binding
        self.disabled_bindings: Set[str] = set()

        # task_name that have been completed
        self.completed_tasks: Set[str] = set()

        self.group_to_automaton: Dict[str, nx.DiGraph] = {}

        # agent_type → [Agent list]
        self.agents_by_type: Dict[str, List[Agent]] = {}

    def register_group(self, group_key: str, tasks: List[str], group_type: str = "drone"):
        """Register a group and allowed agent types for that group."""
        for task in tasks:
            self.task_to_group[task] = group_key
            self.group_to_tasks[group_key].add(task)
        self.group_types[group_key] = {group_type} if isinstance(group_type, str) else set(group_type)

    def disable_binding_for(self, task_name: str):
        """Explicitly allow any agent for this task (no binding enforced)."""
        self.disabled_bindings.add(task_name)

    def record_assignment(self, task_name: str, agent: Agent, agent_type: str) -> bool:
        """
        Called during allocation. Enforces and records group bindings.
        Returns True if assignment is valid or binding-free; False if invalid.
        """
        agent_id = agent.label

        # Allow environment-induced APs to bypass binding checks
        if task_name.startswith("p_found_") or task_name.startswith("p_notfound_"):
            return True

        if task_name in self.disabled_bindings:
            if self.verbose:
                print(f"[BindingManager] {task_name} is exempt from binding rules.")
            return True

        group = self.task_to_group.get(task_name)
        if not group:
            if self.verbose:
                print(f"[BindingManager] No group for {task_name}, no binding needed.")
            return True  # no binding needed

        allowed_types = self.group_types.get(group, set())
        if agent_type not in allowed_types:
            if self.verbose:
                print(f"[BindingManager] {agent_type} not allowed for group {group} of {task_name}")
            return False  # agent type not allowed for this group

        if group not in self.bindings:
            self.bindings[group] = {}

        current_binding = self.bindings[group].get(agent_type)

        if self.verbose:
            print(f"[BindingManager] Group: {group}, Task: {task_name}, Agent Type: {agent_type}, "
                  f"Current: {current_binding}, Proposed: {agent_id}")

        if current_binding is None:
            self.bindings[group][agent_type] = agent_id
            if self.verbose:
                print(f"[BindingManager] Binding {agent_id} to group {group}")
            return True
        else:
            if current_binding == agent_id:
                if self.verbose:
                    print(f"[BindingManager] {agent_id} matches existing binding for {group}")
                return True
            else:
                if self.verbose:
                    print(f"[BindingManager] {agent_id} rejected — group {group} already bound to {current_binding}")
                return False

    def mark_completed(self, task_name: str, labeler=None):
        self.completed_tasks.add(task_name)
        group = self.task_to_group.get(task_name)
        if not group:
            return

        if labeler and group in self.group_to_automaton:
            dfa = self.group_to_automaton[group]
            dfa_state = labeler.states.get(group)
            if dfa_state is not None and labeler._is_accepting(dfa, dfa_state):
                print(f"[BindingManager] DFA for group {group} is accepting — releasing binding.")
                self.bindings.pop(group, None)
            else:
                print(f"[BindingManager] DFA for group {group} not accepting yet — keep binding.")
        else:
            group_tasks = self.group_to_tasks[group]
            if group_tasks.issubset(self.completed_tasks):
                print(f"[BindingManager] All tasks done in group {group} — releasing binding.")
                self.bindings.pop(group, None)

    def get_bound_agent(self, task_name: str, agent_type: str) -> Optional[str]:
        group = self.task_to_group.get(task_name)
        if not group:
            return None
        return self.bindings.get(group, {}).get(agent_type)

    def get_binding_group(self, task_name: str) -> Optional[str]:
        """Return the group associated with a task (AP)."""
        return self.task_to_group.get(task_name)

    def get_bound_agent_for_group(self, group: str, agent_type: str) -> Optional[Agent]:
        """
        Return the 'Agent' object bound to this group and agent_type, if any.
        """
        agent_id = self.bindings.get(group, {}).get(agent_type)
        if agent_id is None:
            return None

        # Look up the actual Agent object by ID
        for agent in self.agents_by_type.get(agent_type, []):
            if agent.label == agent_id:
                return agent

        return None

    def get_bound_agent_for_group_by_role(self, group: str, agent_type: str) -> Optional[str]:
        """Return the agent of a given type currently bound to this group, if any."""
        return self.bindings.get(group, {}).get(agent_type)

    def reset(self):
        """Clear all binding state and completed task info."""
        self.bindings.clear()
        self.completed_tasks.clear()

    def get_next_unfinished_task(self, group, labeler):
        """Return the next task in the group that is not yet completed."""
        ordered_tasks = labeler.get_group_ordered_tasks(group)
        for task in ordered_tasks:
            if task not in self.completed_tasks:
                return task
        return None

