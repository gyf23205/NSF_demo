from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import math
import numpy as np

from ltl_core.agent import Agent
from ltl_core.specification import get_ap_prefix, is_environment_ap, Specification
from ltl_core.binding_manager import BindingManager
from ltl_core.labeler import Labeler
from ltl_core.workspace import Workspace
from ltl_core.value_fn import ValueBank

from rl.value_features import build_s_vector

ROLE_TO_TYPE = {"drones": 1, "gvs": 2, "humans": 3}

class RLAllocator:
    def __init__(
        self,
        spec: Specification,
        agents_by_type: Dict[str, List[Agent]],
        binding_manager: BindingManager,
        labeler: Labeler,
        workspace: Workspace,
        *,
        value_bank: ValueBank,
        eta_weight: float = 1.0,
        dv_weight: float = 0.0,
    ):
        self.spec = spec
        self.agents_by_type = agents_by_type
        self.binding_manager = binding_manager
        self.labeler = labeler
        self.ws = workspace
        self.value_bank = value_bank
        self.eta_weight = float(eta_weight)
        self.dv_weight = float(dv_weight)

    @staticmethod
    def _parse_tid(ap: str) -> int:
        try:
            return int(ap.split("_")[2])
        except Exception:
            return -1

    def _eta_pos(self, agent: Agent, tid: int, use_goal: bool = True) -> float:
        if tid < 0 or tid >= len(getattr(self.ws, "target_locations", [])):
            return 1e9
        try:
            ax, ay = (agent.goal[:2] if (use_goal and getattr(agent, "goal", None) is not None)
                      else agent.pos[:2])
            tx, ty = self.ws.target_locations[int(tid)]
            dist = math.hypot(float(tx) - float(ax), float(ty) - float(ay))
            vmax = float(getattr(agent, "max_speed", 1.0)) or 1.0
            return dist / max(vmax, 1e-6)
        except Exception:
            return 1e9

    def _role_for_ap(self, ap: str) -> str:
        return self.spec.get_required_role_by_ap(ap)  # 'drones'/'gvs'/'humans' or 'unknown'

    # ---------- binding helpers (compatible with your BM variants) ------------
    def _is_group_role_bound(self, group: str, role: str) -> bool:
        bm = self.binding_manager
        # try the explicit helpers if present
        if hasattr(bm, "get_bound_agent_for_group_by_role"):
            return bm.get_bound_agent_for_group_by_role(group, role) is not None
        if hasattr(bm, "get_bound_agent_for_group"):
            # some versions expect numeric type; try both
            atype = ROLE_TO_TYPE.get(role, role)
            try:
                return bm.get_bound_agent_for_group(group, atype) is not None
            except Exception:
                return bm.get_bound_agent_for_group(group, role) is not None
        # fallback to internal dicts if exposed
        if hasattr(bm, "bindings"):
            cur = bm.bindings.get(group, {})
            return (role in cur) or (ROLE_TO_TYPE.get(role) in cur)
        return False

    # ------------------------ candidate construction --------------------------
    def _grouped_candidates(
        self, unlocked: Set[str], completed: Set[str], aps_now: Set[str]
    ) -> Dict[str, List[Tuple[str, str]]]:
        bm, lbl = self.binding_manager, self.labeler
        raw: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for ap in unlocked:
            if ap in completed or ap in aps_now or is_environment_ap(ap):
                continue
            role = self._role_for_ap(ap)
            if role == "unknown":
                continue
            group = getattr(bm, "task_to_group", {}).get(ap)
            if not group:
                continue
            if self._is_group_role_bound(group, role):
                continue
            raw[role].append((ap, group))

        per_role: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for role, items in raw.items():
            by_group: Dict[str, List[str]] = defaultdict(list)
            for ap, g in items:
                by_group[g].append(ap)
            for g, aps in by_group.items():
                ordered = lbl.get_group_ordered_tasks(g) or []
                next_task = None
                for t in ordered:
                    if (t in aps) and (t not in completed):
                        next_task = t
                        break
                if next_task is not None:
                    per_role[role].append((next_task, g))
        return per_role

    # ------------------------------- API --------------------------------------
    def choose_eta(
        self, unlocked: Set[str], completed: List[str], aps: Set[str]
    ) -> Dict[Agent, str]:
        completed_set = set(completed)
        actions: Dict[Agent, str] = {}

        cand = self._grouped_candidates(unlocked, completed_set, aps)

        def pick_for_role(role: str) -> None:
            items = cand.get(role, [])
            if not items:
                return
            free_agents = [a for a in self.agents_by_type.get(role, []) if a not in actions]
            if not free_agents:
                return

            best: Optional[Tuple[float, Agent, str]] = None
            for agent in free_agents:
                for ap, group in items:
                    tid = self._parse_tid(ap)
                    q_node = self.labeler.states.get(group)
                    s_vec = build_s_vector(self.ws, ap)
                    v = float(self.value_bank.value_leaf(ap, q_node, s_vec))
                    eta = self._eta_pos(agent, tid, use_goal=True)
                    score = v - self.eta_weight * eta  # dv term unused by default
                    if (best is None) or (score > best[0]):
                        best = (score, agent, ap)

            if best is None:
                return

            _, agent, ap = best
            # ---- FIX: use (task_name, agent_id, agent_type) signature ----
            agent_id = getattr(agent, "id", None)
            agent_type = ROLE_TO_TYPE.get(role, role)
            ok = False
            try:
                ok = self.binding_manager.record_assignment(ap, agent_id, agent_type)
            except TypeError:
                # Fallback if BM uses named args but same signature
                ok = self.binding_manager.record_assignment(task_name=ap, agent_id=agent_id, agent_type=agent_type)
            if ok:
                actions[agent] = ap
            # else: binding rejected; skip

        pick_for_role("drones")
        pick_for_role("gvs")
        pick_for_role("humans")
        return actions
