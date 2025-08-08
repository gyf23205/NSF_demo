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

        # Human - assigned symbolic function check (RESPECT BINDING)
        for agent in self.agents_by_type.get("human", []):
            task = agent.current_symbolic_task
            if task and task in unlocked and task not in completed and task not in aps:
                agent_type = self.spec.get_required_role_by_ap(task)
                # only keep it if binding rules allow it
                if self.binding_manager.record_assignment(task, agent, agent_type):
                    actions[agent] = task
                    assigned.add(agent)
                else:
                    # binding for that group is owned by someone else; drop the stale task
                    agent.reset_symbolic()

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

                    # BLOCK: GV should not take new pickup if already assigned dropoff
                    # if agent_type == "gv" and agent.label in busy_gvs and task.startswith("p_pickup_"):
                        # continue

                    success = self.binding_manager.record_assignment(task, agent, agent_type)
                    if not success:
                        continue

                    actions[agent] = task
                    assigned.add(agent)
                    break  # one agent per group/type
            # break  # one task per group
        return actions

