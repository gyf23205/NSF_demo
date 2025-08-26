# rl/allocator_rl.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional
import math

from ltl_core.agent import Agent
from ltl_core.specification import get_ap_prefix, AP_TYPE_PREFIX_MAP, Specification
from ltl_core.binding_manager import BindingManager
from ltl_core.labeler import Labeler
from ltl_core.workspace import Workspace
from ltl_core.value_fn import ValueBank
from rl.value_features import build_s_vector


class RLAllocator:
    """
    Random-like allocator with a value-aware tie-break on preemptible physical tasks:
      - One AP per group per tick.
      - Humans (symbolic) are sticky; we re-record the same assignment each tick so progress accumulates.
      - Drones: NAV is preemptible, SCAN continues.
      - GVs   : PICKUP is preemptible, DROPOFF continues.
      - Among same-priority preemptible tasks, ValueBank ranks tasks; agent choice remains nearest-by-ETA.
    """

    def __init__(
        self,
        spec: Specification,
        agents_by_type: Dict[str, List[Agent]],
        binding_manager: BindingManager,
        labeler: Labeler,
        workspace: Workspace,
        *,
        value_bank: Optional[ValueBank] = None,
        eta_weight: float = 1.0,
        dv_weight: float = 0.0,
    ):
        self.spec = spec
        self.agents_by_type = agents_by_type
        self.binding_manager = binding_manager
        self.labeler = labeler
        self.workspace = workspace

        self.value_bank = value_bank
        self.eta_weight = float(eta_weight)
        self.dv_weight = float(dv_weight)

    # --------------------------- helpers ---------------------------

    @staticmethod
    def _parse_tid(ap: str) -> int:
        try:
            return int(ap.split("_")[2])
        except Exception:
            return 0

    @staticmethod
    def _dist(p, q) -> float:
        return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))

    def _target_pos_for_task(self, ap: str):
        """Return (x, y) for this AP's physical target."""
        pref = get_ap_prefix(ap)
        tid = self._parse_tid(ap)
        if pref == "p_dropoff":
            locs = getattr(self.workspace, "dropoff_locations", {})
            return tuple(locs.get(tid, (0.0, 0.0)))
        else:
            locs = getattr(self.workspace, "target_locations", [])
            if 0 <= tid < len(locs):
                return tuple(locs[tid])
        return (0.0, 0.0)

    def _eta_sec(self, agent: Agent, ap: str) -> float:
        """ETA proxy: distance / speed, using agent.goal when present."""
        pos = tuple(agent.goal[:2]) if getattr(agent, "goal", None) is not None else tuple(agent.pos[:2])
        tgt = self._target_pos_for_task(ap)
        vmax = float(getattr(agent, "max_speed", 1.0)) or 1.0
        return self._dist(pos, tgt) / max(vmax, 1e-6)

    def _value_of(self, ap: str, group: str) -> float:
        """Leaf value for the AP using current DFA state and s-vector."""
        if self.value_bank is None:
            return 0.0
        q = self.labeler.states.get(group)
        s = build_s_vector(self.workspace, ap)
        try:
            return float(self.value_bank.value_leaf(ap, q, s))
        except Exception:
            return 0.0

    # --------------------------- main policy ----------------------------

    def choose_eta(
        self, unlocked: Set[str], completed: List[str], aps: Set[str]
    ) -> Dict[Agent, str]:
        completed_set = set(completed)
        actions: Dict[Agent, str] = {}

        assigned_agents: Set[Agent] = set()
        claimed_groups: Set[str] = set()

        # ---- 0) Human symbolic continuation (sticky with re-record) ----
        for h in self.agents_by_type.get("humans", []):
            task = getattr(h, "current_symbolic_task", None)
            if not task:
                continue
            if (task in unlocked) and (task not in completed_set) and (task not in aps):
                role = self.spec.get_required_role_by_ap(task)
                group = getattr(self.binding_manager, "task_to_group", {}).get(task)
                if group in claimed_groups:
                    # already satisfied by the group this tick; still add action for timeline
                    actions[h] = task
                    assigned_agents.add(h)
                    continue
                ok = False
                try:
                    ok = self.binding_manager.record_assignment(task, h, role)
                except TypeError:
                    ok = self.binding_manager.record_assignment(task_name=task, agent=h, agent_type=role)
                if ok:
                    actions[h] = task
                    assigned_agents.add(h)
                    if group:
                        claimed_groups.add(group)
                # Do not reset symbolic on failure; keep previous progress/state.
            else:
                # task no longer eligible → allow reallocation from scratch
                if hasattr(h, "reset_symbolic"):
                    h.reset_symbolic()

        # ---- 1) Non-preemptible physical continuation (SCAN/DROPOFF) ----
        def _continue_non_preemptible(agent_type: str):
            for group in getattr(self.binding_manager, "group_to_tasks", {}):
                if group in claimed_groups:
                    continue
                try:
                    bound = self.binding_manager.get_bound_agent_for_group(group, agent_type=agent_type)
                except Exception:
                    bound = None
                if bound is None or bound in assigned_agents:
                    continue

                ordered = self.labeler.get_group_ordered_tasks(group) or []
                for t in ordered:
                    if not (t in unlocked and t not in completed_set and t not in aps):
                        continue
                    if self.spec.get_required_role_by_ap(t) != agent_type:
                        continue
                    pref = get_ap_prefix(t)
                    # skip preemptible in this pass
                    if agent_type == "drones" and pref == "p_nav":
                        break
                    if agent_type == "gvs" and pref == "p_pickup":
                        break
                    ok = False
                    try:
                        ok = self.binding_manager.record_assignment(t, bound, agent_type)
                    except TypeError:
                        ok = self.binding_manager.record_assignment(task_name=t, agent=bound, agent_type=agent_type)
                    if ok:
                        actions[bound] = t
                        assigned_agents.add(bound)
                        claimed_groups.add(group)
                    break

        _continue_non_preemptible("drones")
        _continue_non_preemptible("gvs")

        # ---- 2) Collect remaining candidates ----
        pending_phys: List[Tuple[int, str, str, str, int, str]] = []
        pending_symb: List[Tuple[str, str]] = []

        for group, _ in getattr(self.binding_manager, "group_to_tasks", {}).items():
            if group in claimed_groups:
                continue
            ordered = self.labeler.get_group_ordered_tasks(group) or []
            for t in ordered:
                if group in claimed_groups:
                    break
                if (t not in unlocked) or (t in completed_set) or (t in aps):
                    continue
                role = self.spec.get_required_role_by_ap(t)
                pref = get_ap_prefix(t)
                ap_kind = AP_TYPE_PREFIX_MAP.get(pref, "physical")

                if ap_kind == "physical":
                    try:
                        tid = int(t.split("_")[2])
                    except Exception:
                        tid = 0
                    pr = self.workspace.get_target_priority(tid) if getattr(self.workspace, "get_target_priority", None) else 0
                    pending_phys.append((pr, pref, group, t, tid, role))
                else:
                    pending_symb.append((group, t))

        # ---- 3) Split/Order physical tasks ----
        drone_tasks: List[Tuple[int, str, str, str, int, str]] = []
        gv_pickups: List[Tuple[int, str, str, str, int, str]] = []
        gv_dropoffs: List[Tuple[int, str, str, str, int, str]] = []

        for pr, pref, group, t, tid, role in pending_phys:
            if role == "drones":
                drone_tasks.append((pr, pref, group, t, tid, role))
            elif role == "gvs":
                if pref == "p_pickup":
                    gv_pickups.append((pr, pref, group, t, tid, role))
                else:
                    gv_dropoffs.append((pr, pref, group, t, tid, role))

        def _sort_preemptible(tasks: List[Tuple[int, str, str, str, int, str]]):
            # Sort by (priority desc, value desc) to respect tie-breaker rule.
            tasks.sort(key=lambda x: (x[0], self._value_of(x[3], x[2])), reverse=True)

        _sort_preemptible(drone_tasks)   # includes NAV (preemptible)
        _sort_preemptible(gv_pickups)    # PICKUP (preemptible)
        # gv_dropoffs remain in natural order (continuation usually covers them)

        # ---- 4) Greedy assignment for physical tasks ----
        def _assign_physical(tasks: List[Tuple[int, str, str, str, int, str]], agent_type: str):
            free_pool: List[Agent] = [a for a in self.agents_by_type.get(agent_type, []) if a not in assigned_agents]
            if not free_pool:
                return
            for pr, pref, group, task, tid, role in list(tasks):
                if group in claimed_groups:
                    continue

                preemptible = (agent_type == "drones" and pref == "p_nav") or (agent_type == "gvs" and pref == "p_pickup")

                # Prefer continuation agent for non-preemptible steps
                bound: Optional[Agent] = None
                try:
                    bound = self.binding_manager.get_bound_agent_for_group(group, agent_type=agent_type)
                except Exception:
                    bound = None

                candidate_list: List[Agent] = []
                if (not preemptible) and (bound is not None) and (bound in free_pool) and (bound not in assigned_agents):
                    candidate_list = [bound]
                else:
                    # Choose nearest agent by ETA for this task
                    free_sorted = sorted(
                        [a for a in free_pool if a not in assigned_agents],
                        key=lambda a: self._eta_sec(a, task)
                    )
                    candidate_list = free_sorted

                # Try candidates in order until a binding sticks
                chosen = None
                for a in candidate_list:
                    ok = False
                    try:
                        ok = self.binding_manager.record_assignment(task, a, agent_type)
                    except TypeError:
                        ok = self.binding_manager.record_assignment(task_name=task, agent=a, agent_type=agent_type)
                    if ok:
                        chosen = a
                        break

                if chosen is not None:
                    actions[chosen] = task
                    assigned_agents.add(chosen)
                    if chosen in free_pool:
                        free_pool.remove(chosen)
                    claimed_groups.add(group)
                    if not free_pool:
                        break  # no more agents of this type

        _assign_physical(drone_tasks, "drones")
        _assign_physical(gv_pickups, "gvs")
        _assign_physical(gv_dropoffs, "gvs")

        # ---- 5) Assign new symbolic tasks (start) ----
        free_humans: List[Agent] = [h for h in self.agents_by_type.get("humans", []) if h not in assigned_agents]
        for group, t in pending_symb:
            if group in claimed_groups:
                continue
            # prefer an already bound human, if free
            bound_h: Optional[Agent] = None
            try:
                bound_h = self.binding_manager.get_bound_agent_for_group(group, agent_type="humans")
            except Exception:
                bound_h = None

            target_h: Optional[Agent] = None
            if (bound_h is not None) and (bound_h not in assigned_agents):
                target_h = bound_h
            elif free_humans:
                target_h = free_humans.pop(0)
            if target_h is None:
                continue

            ok = False
            try:
                ok = self.binding_manager.record_assignment(t, target_h, "humans")
            except TypeError:
                ok = self.binding_manager.record_assignment(task_name=t, agent=target_h, agent_type="humans")
            if ok:
                actions[target_h] = t
                assigned_agents.add(target_h)
                claimed_groups.add(group)

        return actions
