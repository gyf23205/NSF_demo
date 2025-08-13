from .agent import Agent
from .specification import get_ap_prefix, AP_TYPE_PREFIX_MAP
from typing import Dict, Set, List


class RandomAllocator:
    def __init__(self, spec, agents_by_type, binding_manager, labeler, workspace=None):
        self.spec = spec
        self.agents_by_type = agents_by_type
        self.binding_manager = binding_manager
        self.labeler = labeler
        self.workspace = workspace

    def _target_pos_for_task(self, ap: str):
        """Return (x,y) in grid for this AP's physical target."""
        tid = int(ap.split("_")[2])
        pref = get_ap_prefix(ap)
        if pref == "p_dropoff":
            return self.workspace.dropoff_locations[tid]
        return self.workspace.target_locations[tid]

    @staticmethod
    def _dist(pxy, qxy):
        dx = float(pxy[0]) - float(qxy[0])
        dy = float(pxy[1]) - float(qxy[1])
        return (dx*dx + dy*dy) ** 0.5

    def choose(self, unlocked: Set[str], completed: List[str], aps: Set[str]) -> Dict[Agent, str]:
        actions = {}
        assigned = set()

        # Human - assigned symbolic function check (RESPECT BINDING)
        for agent in self.agents_by_type.get("humans", []):
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
                    # if agent_type == "gvs" and agent.label in busy_gvs and task.startswith("p_pickup_"):
                        # continue

                    success = self.binding_manager.record_assignment(task, agent, agent_type)
                    if not success:
                        continue

                    actions[agent] = task
                    assigned.add(agent)
                    break  # one agent per group/type
            # break  # one task per group
        return actions
    
    def choose_priority(self, unlocked: Set[str], completed: List[str], aps: Set[str]) -> Dict["Agent", str]:
        actions: Dict["Agent", str] = {}
        assigned = set()
        claimed_tasks = set()  # avoid double-assigning the same AP within this call

        # 0) Keep any in-progress human symbolic tasks that are still valid (binding-respecting)
        for agent in self.agents_by_type.get("humans", []):
            task = getattr(agent, "current_symbolic_task", None)
            if task and (task in unlocked) and (task not in completed) and (task not in aps):
                agent_type = self.spec.get_required_role_by_ap(task)
                if self.binding_manager.record_assignment(task, agent, agent_type):
                    actions[agent] = task
                    assigned.add(agent)
                    claimed_tasks.add(task)
                else:
                    if hasattr(agent, "reset_symbolic"):
                        agent.reset_symbolic()

        # === 1) CONTINUATION PASS (finish non-preemptible steps first) ===
        # - Drones: do NOT continue p_nav_* (preemptible); DO continue p_scan_* (non-preemptible)
        # - GVs:    do NOT continue p_pickup_* (preemptible); DO continue p_dropoff_* (non-preemptible)
        def _assign_group_continuation(agent_type: str):
            for group in self.binding_manager.group_to_tasks:
                try:
                    bound_agent = self.binding_manager.get_bound_agent_for_group(group, agent_type=agent_type)
                except Exception:
                    bound_agent = None
                if bound_agent is None or bound_agent in assigned:
                    continue

                ordered = self.labeler.get_group_ordered_tasks(group) or []
                for t in ordered:
                    if t in claimed_tasks:
                        continue
                    if not (t in unlocked and t not in completed and t not in aps):
                        continue

                    req = self.spec.get_required_role_by_ap(t)
                    if req != agent_type:
                        continue

                    pref = get_ap_prefix(t)
                    # Preemption rules for continuation:
                    if agent_type == "drones" and pref == "p_nav":
                        break  # let global priority handle nav (preemptible)
                    if agent_type == "gvs" and pref == "p_pickup":
                        break  # let global priority handle pickup (preemptible)

                    # Non-preemptible (e.g., p_scan_* for drones, p_dropoff_* for GVs) -> continue with bound agent
                    if self.binding_manager.record_assignment(t, bound_agent, agent_type):
                        actions[bound_agent] = t
                        assigned.add(bound_agent)
                        claimed_tasks.add(t)
                    break  # do not look further in the group's sequence

        _assign_group_continuation("drones")
        _assign_group_continuation("gvs")

        # 2) Collect remaining pending tasks across ALL groups (skip any already claimed)
        pending_phys = []   # list of tuples: (prio, pref, group, task, tid, agent_type)
        pending_symb = []   # symbolic tasks we may attempt after physicals
        for group in self.binding_manager.group_to_tasks:
            tasks = self.labeler.get_group_ordered_tasks(group) or []
            for t in tasks:
                if t in claimed_tasks:
                    continue
                if (t not in unlocked) or (t in completed) or (t in aps):
                    continue

                pref = get_ap_prefix(t)
                ap_kind = AP_TYPE_PREFIX_MAP.get(pref)
                agent_type = self.spec.get_required_role_by_ap(t)

                if ap_kind == "physical":
                    # target index assumed at position 2 in AP string: p_xxx_<tid>_...
                    try:
                        tid = int(t.split("_")[2])
                    except Exception:
                        tid = 0
                    pr = self.workspace.get_target_priority(tid) if getattr(self, "workspace", None) else 0
                    pending_phys.append((pr, pref, group, t, tid, agent_type))
                else:
                    pending_symb.append((group, t))

        # 3) Split physical tasks into buckets; sort preemptible buckets by priority (desc)
        drone_tasks = []     # includes p_nav_* (preemptible) and any drone-physical others if present
        gv_pickups = []      # p_pickup_* (preemptible)
        gv_dropoffs = []     # p_dropoff_* (non-preemptible; handled last)
        for pr, pref, group, t, tid, agent_type in pending_phys:
            if agent_type == "drones":
                drone_tasks.append((pr, pref, group, t, tid, agent_type))
            elif agent_type == "gvs":
                if pref == "p_dropoff":
                    gv_dropoffs.append((pr, pref, group, t, tid, agent_type))
                else:
                    gv_pickups.append((pr, pref, group, t, tid, agent_type))

        drone_tasks.sort(key=lambda x: x[0], reverse=True)   # high → low priority
        gv_pickups.sort(key=lambda x: x[0], reverse=True)    # high → low priority
        # gv_dropoffs: order not priority-driven; keep as given

        # 4) Physical assignment helper (preemptible travel ignores binding; non-preemptible respects it)
        def _assign_physical(task_list, pool_role):
            free_pool = [a for a in self.agents_by_type.get(pool_role, []) if a not in assigned]

            for pr, pref, group, task, tid, agent_type in task_list:
                if task in claimed_tasks or agent_type != pool_role:
                    continue

                # Prefer/ignore binding based on preemptibility:
                # - Drones: p_nav is preemptible; p_scan is NOT
                # - GVs:    p_pickup is preemptible; p_dropoff is NOT
                preemptible = ((pool_role == "drones" and pref == "p_nav") or
                            (pool_role == "gvs"    and pref == "p_pickup"))

                try:
                    bound_agent = self.binding_manager.get_bound_agent_for_group(group, agent_type=agent_type)
                except Exception:
                    bound_agent = None

                candidates = []
                if (not preemptible) and (bound_agent is not None) and (bound_agent in free_pool) and (bound_agent not in assigned):
                    # Non-preemptible: keep the bound agent
                    candidates = [bound_agent]
                else:
                    # Preemptible OR no usable binding: pick nearest free agent
                    tgt = self._target_pos_for_task(task)
                    free_pool.sort(key=lambda a: self._dist(a.pos[:2], tgt))
                    candidates = list(free_pool)

                for agent in candidates:
                    if self.binding_manager.record_assignment(task, agent, agent_type):
                        actions[agent] = task
                        assigned.add(agent)
                        claimed_tasks.add(task)
                        if agent in free_pool:
                            free_pool.remove(agent)
                        break  # move to next task

        # 5) Assign physical tasks with cross-group priority (dropoffs last)
        _assign_physical(drone_tasks, "drones")
        _assign_physical(gv_pickups, "gvs")
        _assign_physical(gv_dropoffs, "gvs")

        # 6) Symbolic fallback (binding preference, no preemption concept)
        for group, t in pending_symb:
            if t in claimed_tasks:
                continue
            agent_type = self.spec.get_required_role_by_ap(t)
            if agent_type is None:
                continue

            try:
                bound_agent = self.binding_manager.get_bound_agent_for_group(group, agent_type=agent_type)
            except Exception:
                bound_agent = None

            if (bound_agent is not None) and (bound_agent not in assigned):
                cand = [bound_agent]
            else:
                cand = [a for a in self.agents_by_type.get(agent_type, []) if a not in assigned]

            for agent in cand:
                if self.binding_manager.record_assignment(t, agent, agent_type):
                    actions[agent] = t
                    assigned.add(agent)
                    claimed_tasks.add(t)
                    break

        return actions
