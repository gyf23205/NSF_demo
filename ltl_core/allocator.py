from .agent import Agent
from typing import Dict, Set, List


class RandomAllocator:
    def __init__(self, spec, agents_by_type, binding_manager, labeler):
        self.spec = spec
        self.agents_by_type = agents_by_type
        self.binding_manager = binding_manager
        self.labeler = labeler

    def choose(self, unlocked: Set[str], completed: List[str], aps: Set[str]) -> Dict[Agent, str]:
        actions = {}
        assigned = set()

        for group in self.binding_manager.group_to_tasks:
            tasks = self.labeler.get_group_ordered_tasks(group)
            if not tasks:
                continue

            for task in tasks:
                if task not in unlocked or task in completed or task in aps:
                    continue

                agent_type = self.spec.get_required_role_by_ap(task)
                if agent_type is None:
                    continue

                for agent in self.agents_by_type.get(agent_type, []):
                    if agent in assigned:
                        continue
                    # print(f"[ALLOCATOR] Trying to assign {agent.label} to task {task}")
                    success = self.binding_manager.record_assignment(task, agent, agent_type)
                    # print(f"[ALLOCATOR] → Success: {success}")
                    if not success:
                        continue
                    actions[agent] = task
                    assigned.add(agent)
                    break  # only one agent per group/type
                break  # only assign one task per group
        return actions

