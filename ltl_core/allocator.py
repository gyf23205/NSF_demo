from .agent import Agent
from .specification import get_ap_prefix, AP_TYPE_PREFIX_MAP
from typing import Dict, Set, List
import math


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

    def choose_priority_util(self, unlocked: Set[str], completed: List[str], aps: Set[str]) -> Dict["Agent", str]:
        actions: Dict["Agent", str] = {}
        assigned = set()
        claimed_tasks = set()  # avoid double-assigning within this call

        UTIL_THRESH = 60  # percent

        def human_util(a) -> float:
            # Missing attr -> treat as not overloaded
            u = getattr(a, "utilization", 0)
            try:
                return float(u)
            except Exception:
                return 0.0

        def _autonomy_alt(ap: str) -> str | None:
            """If ap is a human symbolic OR-branch we can offload (priority/triage),
            return the drone variant (swap _3_ → _1_). Else None."""
            pref = get_ap_prefix(ap)
            if pref not in ("p_priority", "p_triage"):
                return None
            parts = ap.split("_")
            if len(parts) >= 6 and parts[3] == "3":
                parts[3] = "1"
                return "_".join(parts[:6])
            return None

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
        pending_phys = []   # tuples: (prio, pref, group, task, tid, agent_type)
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
        # gv_dropoffs: order not priority-driven

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

        # 6) Symbolic fallback — NOW human-utilization aware (+ optional drone offload)
        #    Strategy per symbolic task t:
        #    - If t requires humans:
        #        * Prefer bound human if NOT overloaded.
        #        * Else prefer another non-overloaded human (lowest utilization).
        #        * Else, if t is p_priority/p_triage and the drone-variant is available, assign that to a drone.
        #        * Else, as last resort, assign the least-overloaded human (avoid deadlock).
        #    - If t requires drones/humans otherwise, keep previous behavior.
        pending_symb_set = set(t for _, t in pending_symb)

        for group, t in pending_symb:
            if t in claimed_tasks:
                continue

            role = self.spec.get_required_role_by_ap(t)
            if role is None:
                continue

            # Humans: apply utilization-aware selection and OR-branch offload
            if role == "humans":
                # 1) usable bound human?
                try:
                    bound_h = self.binding_manager.get_bound_agent_for_group(group, agent_type="humans")
                except Exception:
                    bound_h = None

                # build human pool sorted by utilization (low → high)
                free_humans = [h for h in self.agents_by_type.get("humans", []) if h not in assigned]
                free_humans.sort(key=human_util)

                # split by overload
                ok_humans = [h for h in free_humans if human_util(h) < UTIL_THRESH]
                overloaded = [h for h in free_humans if h not in ok_humans]

                # case A: bound human and not overloaded → take it
                if (bound_h is not None) and (bound_h not in assigned) and (human_util(bound_h) < UTIL_THRESH):
                    cand = [bound_h]
                else:
                    # case B: any non-overloaded human?
                    cand = ok_humans if ok_humans else []

                picked = False
                for agent in cand:
                    if self.binding_manager.record_assignment(t, agent, "humans"):
                        actions[agent] = t
                        assigned.add(agent)
                        claimed_tasks.add(t)
                        picked = True
                        break

                if picked:
                    continue  # next symbolic

                # case C: try to offload to drone via OR-branch (priority/triage only)
                alt = _autonomy_alt(t)
                if alt and (alt in pending_symb_set) and (alt in unlocked) and (alt not in completed) and (alt not in aps):
                    # pick any free drone (symbolic; no distance notion)
                    free_drones = [d for d in self.agents_by_type.get("drones", []) if d not in assigned]
                    for d in free_drones:
                        if self.binding_manager.record_assignment(alt, d, "drones"):
                            actions[d] = alt
                            assigned.add(d)
                            claimed_tasks.add(alt)
                            claimed_tasks.add(t)  # suppress human variant this tick
                            picked = True
                            break
                    if picked:
                        continue  # next symbolic

                # case D: last resort — assign least-overloaded human if exists
                fallback_pool = ([bound_h] if (bound_h is not None and bound_h not in assigned) else []) + overloaded
                # de-dup while preserving order
                seen = set()
                fallback_pool = [h for h in fallback_pool if (h not in seen and not seen.add(h))]
                for agent in fallback_pool:
                    if self.binding_manager.record_assignment(t, agent, "humans"):
                        actions[agent] = t
                        assigned.add(agent)
                        claimed_tasks.add(t)
                        picked = True
                        break
                # If still not picked: nothing to do for this t this tick
                continue

            # Non-human symbolic (rare) — keep previous behavior
            try:
                bound_agent = self.binding_manager.get_bound_agent_for_group(group, agent_type=role)
            except Exception:
                bound_agent = None

            if (bound_agent is not None) and (bound_agent not in assigned):
                cand = [bound_agent]
            else:
                cand = [a for a in self.agents_by_type.get(role, []) if a not in assigned]

            for agent in cand:
                if self.binding_manager.record_assignment(t, agent, role):
                    actions[agent] = t
                    assigned.add(agent)
                    claimed_tasks.add(t)
                    break

        return actions
    
    def choose_eta(self, unlocked: Set[str], completed: List[str], aps: Set[str]) -> Dict["Agent", str]:
        actions: Dict["Agent", str] = {}
        assigned = set()
        claimed_tasks = set()  # avoid double-assigning within this call

        UTIL_THRESH = 60  # percent

        # --------- helpers (lightweight, stored on self for persistence) ----------
        def human_util(a) -> float:
            u = getattr(a, "utilization", 0)
            try:
                return float(u)
            except Exception:
                return 0.0

        def _autonomy_alt(ap: str) -> str | None:
            """If ap is a human symbolic OR-branch (priority/triage), return the drone variant (_3_ → _1_)."""
            pref = get_ap_prefix(ap)
            if pref not in ("p_priority", "p_triage"):
                return None
            parts = ap.split("_")
            if len(parts) >= 6 and parts[3] == "3":
                parts[3] = "1"
                return "_".join(parts[:6])
            return None

        def _eta_to_tid(agent, tid: int) -> float:
            """ETA proxy from agent.goal (if exists) else from position to target tid."""
            if not getattr(self, "workspace", None):
                return 1e9
            try:
                gx, gy = (agent.goal[:2] if getattr(agent, "goal", None) is not None else agent.pos[:2])
                tx, ty = self.workspace.target_locations[int(tid)]
            except Exception:
                return 1e9
            dx = float(tx) - float(gx)
            dy = float(ty) - float(gy)
            dist = math.hypot(dx, dy)
            vmax = getattr(agent, "max_speed", None)
            return dist / max(float(vmax or 1.0), 1e-6)

        def _get_sticky(map_name: str, group: str):
            if getattr(self, map_name, None) is None:
                setattr(self, map_name, {})
            return getattr(self, map_name).get(group)

        def _set_sticky(map_name: str, group: str, agent):
            if getattr(self, map_name, None) is None:
                setattr(self, map_name, {})
            getattr(self, map_name)[group] = agent

        # alias names for clarity
        DRONE_STICKY = "_drone_sticky"
        GV_STICKY    = "_gv_sticky"

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
        pending_phys = []   # tuples: (prio, pref, group, task, tid, agent_type)
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

        # 3) Split physical tasks into buckets
        drone_nav   = []    # p_nav_* (preemptible) → auction
        drone_other = []    # non-preemptible drone physical (e.g., p_scan_*) → sticky/nearest
        gv_pickups  = []    # p_pickup_* (preemptible) → auction
        gv_dropoffs = []    # p_dropoff_* (non-preemptible) → continuation / fallback

        for pr, pref, group, t, tid, agent_type in pending_phys:
            if agent_type == "drones":
                if pref == "p_nav":
                    drone_nav.append((pr, pref, group, t, tid, agent_type))
                else:
                    drone_other.append((pr, pref, group, t, tid, agent_type))
            elif agent_type == "gvs":
                if pref == "p_dropoff":
                    gv_dropoffs.append((pr, pref, group, t, tid, agent_type))
                else:
                    gv_pickups.append((pr, pref, group, t, tid, agent_type))

        # Sort preemptible buckets by priority desc (high → low)
        drone_nav.sort(key=lambda x: (-x[0], x[4]))
        gv_pickups.sort(key=lambda x: (-x[0], x[4]))
        # gv_dropoffs / drone_other order is less critical

        # 4-A) DRONE NAV: priority-aware auction with stickiness
        free_drones = [d for d in self.agents_by_type.get("drones", []) if d not in assigned]
        for pr, pref, group, task, tid, agent_type in drone_nav:
            if task in claimed_tasks or not free_drones:
                continue

            sticky = _get_sticky(DRONE_STICKY, group)
            best = None
            best_cost = float("inf")
            for d in list(free_drones):
                cost = _eta_to_tid(d, tid)
                if sticky is not None and d is sticky:
                    cost *= 0.5  # hysteresis (keep same drone on group)
                # mild penalty if switching away from current bound group
                if getattr(d, "bound_group", None) not in (None, group):
                    cost *= 1.15
                if cost < best_cost:
                    best_cost = cost
                    best = d

            if best is not None and self.binding_manager.record_assignment(task, best, "drones"):
                actions[best] = task
                assigned.add(best)
                claimed_tasks.add(task)
                try:
                    free_drones.remove(best)
                except ValueError:
                    pass
                _set_sticky(DRONE_STICKY, group, best)
                setattr(best, "bound_group", group)

        # 4-B) GV PICKUPS: priority-aware auction with stickiness
        free_gvs = [g for g in self.agents_by_type.get("gvs", []) if g not in assigned]
        for pr, pref, group, task, tid, agent_type in gv_pickups:
            if task in claimed_tasks or not free_gvs:
                continue

            sticky = _get_sticky(GV_STICKY, group)
            best = None
            best_cost = float("inf")
            for gv in list(free_gvs):
                cost = _eta_to_tid(gv, tid)
                if sticky is not None and gv is sticky:
                    cost *= 0.5  # hysteresis (keep same GV on group)
                if getattr(gv, "bound_group", None) not in (None, group):
                    cost *= 1.15
                if cost < best_cost:
                    best_cost = cost
                    best = gv

            if best is not None and self.binding_manager.record_assignment(task, best, "gvs"):
                actions[best] = task
                assigned.add(best)
                claimed_tasks.add(task)
                try:
                    free_gvs.remove(best)
                except ValueError:
                    pass
                _set_sticky(GV_STICKY, group, best)
                setattr(best, "bound_group", group)

        # 4-C) DRONE other physical (non-preemptible): prefer sticky drone, else nearest free
        free_drones = [d for d in self.agents_by_type.get("drones", []) if d not in assigned]
        for pr, pref, group, task, tid, agent_type in drone_other:
            if task in claimed_tasks:
                continue
            if not free_drones:
                break

            picked = None
            sticky = _get_sticky(DRONE_STICKY, group)
            # 1) sticky drone available?
            if sticky is not None and sticky in free_drones:
                picked = sticky
            else:
                # 2) nearest free drone to the target
                try:
                    tx, ty = self.workspace.target_locations[int(tid)]
                    free_drones.sort(key=lambda d: math.hypot((d.pos[0]-tx), (d.pos[1]-ty)))
                except Exception:
                    pass
                if free_drones:
                    picked = free_drones[0]

            if picked is not None and self.binding_manager.record_assignment(task, picked, "drones"):
                actions[picked] = task
                assigned.add(picked)
                claimed_tasks.add(task)
                try:
                    free_drones.remove(picked)
                except ValueError:
                    pass
                _set_sticky(DRONE_STICKY, group, picked)
                setattr(picked, "bound_group", group)

        # 4-D) GV DROPOFFS (non-preemptible): let continuation handle most; simple fallback if any left
        free_gvs = [g for g in self.agents_by_type.get("gvs", []) if g not in assigned]
        for pr, pref, group, task, tid, agent_type in gv_dropoffs:
            if task in claimed_tasks or not free_gvs:
                continue
            # Prefer sticky GV if available, else any free GV (dropoffs should be rare here)
            picked = None
            sticky = _get_sticky(GV_STICKY, group)
            if sticky is not None and sticky in free_gvs:
                picked = sticky
            else:
                picked = free_gvs[0]
            if picked is not None and self.binding_manager.record_assignment(task, picked, "gvs"):
                actions[picked] = task
                assigned.add(picked)
                claimed_tasks.add(task)
                try:
                    free_gvs.remove(picked)
                except ValueError:
                    pass
                _set_sticky(GV_STICKY, group, picked)
                setattr(picked, "bound_group", group)

        # 5) Symbolic fallback — human-utilization aware (+ optional drone offload)
        pending_symb_set = set(t for _, t in pending_symb)

        for group, t in pending_symb:
            if t in claimed_tasks:
                continue

            role = self.spec.get_required_role_by_ap(t)
            if role is None:
                continue

            # Humans: apply utilization-aware selection and OR-branch offload
            if role == "humans":
                # 1) usable bound human?
                try:
                    bound_h = self.binding_manager.get_bound_agent_for_group(group, agent_type="humans")
                except Exception:
                    bound_h = None

                # build human pool sorted by utilization (low → high)
                free_humans = [h for h in self.agents_by_type.get("humans", []) if h not in assigned]
                free_humans.sort(key=human_util)

                # split by overload
                ok_humans = [h for h in free_humans if human_util(h) < UTIL_THRESH]
                overloaded = [h for h in free_humans if h not in ok_humans]

                # case A: bound human and not overloaded → take it
                if (bound_h is not None) and (bound_h not in assigned) and (human_util(bound_h) < UTIL_THRESH):
                    cand = [bound_h]
                else:
                    # case B: any non-overloaded human?
                    cand = ok_humans if ok_humans else []

                picked = False
                for agent in cand:
                    if self.binding_manager.record_assignment(t, agent, "humans"):
                        actions[agent] = t
                        assigned.add(agent)
                        claimed_tasks.add(t)
                        picked = True
                        break

                if picked:
                    continue  # next symbolic

                # case C: try to offload to drone via OR-branch (priority/triage only)
                alt = _autonomy_alt(t)
                if alt and (alt in pending_symb_set) and (alt in unlocked) and (alt not in completed) and (alt not in aps):
                    free_drs = [d for d in self.agents_by_type.get("drones", []) if d not in assigned]
                    for d in free_drs:
                        if self.binding_manager.record_assignment(alt, d, "drones"):
                            actions[d] = alt
                            assigned.add(d)
                            claimed_tasks.add(alt)
                            claimed_tasks.add(t)  # suppress human variant this tick
                            picked = True
                            break
                    if picked:
                        continue

                # case D: last resort — assign least-overloaded human if exists (avoid deadlock)
                fallback_pool = ([bound_h] if (bound_h is not None and bound_h not in assigned) else []) + overloaded
                # de-dup while preserving order
                seen = set()
                fallback_pool = [h for h in fallback_pool if (h not in seen and not seen.add(h))]
                for agent in fallback_pool:
                    if self.binding_manager.record_assignment(t, agent, "humans"):
                        actions[agent] = t
                        assigned.add(agent)
                        claimed_tasks.add(t)
                        picked = True
                        break
                continue  # done with human-required symbolic

            # Non-human symbolic (rare) — keep previous behavior
            try:
                bound_agent = self.binding_manager.get_bound_agent_for_group(group, agent_type=role)
            except Exception:
                bound_agent = None

            if (bound_agent is not None) and (bound_agent not in assigned):
                cand = [bound_agent]
            else:
                cand = [a for a in self.agents_by_type.get(role, []) if a not in assigned]

            for agent in cand:
                if self.binding_manager.record_assignment(t, agent, role):
                    actions[agent] = t
                    assigned.add(agent)
                    claimed_tasks.add(t)
                    break

        return actions

