from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import math
import numpy as np

from ltl_core.agent import Agent
from ltl_core.specification import is_environment_ap, Specification
from ltl_core.binding_manager import BindingManager
from ltl_core.labeler import Labeler
from ltl_core.workspace import Workspace
from ltl_core.value_fn import ValueBank

from rl.value_features import build_s_vector


class RLAllocator:
    """
    Value-aware allocator that:
      - assigns greedily across *different* groups,
      - enforces ONE AP PER GROUP PER TICK (all roles),
      - keeps symbolic tasks sticky to the original human ONLY,
      - allows physical reallocation for drones/GVs.
    """

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
        locs = getattr(self.ws, "target_locations", [])
        if tid < 0 or tid >= len(locs):
            return 1e9
        try:
            ax, ay = (agent.goal[:2] if (use_goal and getattr(agent, "goal", None) is not None)
                      else agent.pos[:2])
            tx, ty = locs[int(tid)]
            dist = math.hypot(float(tx) - float(ax), float(ty) - float(ay))
            vmax = float(getattr(agent, "max_speed", 1.0)) or 1.0
            return dist / max(vmax, 1e-6)
        except Exception:
            return 1e9

    def _role_for_ap(self, ap: str) -> str:
        return self.spec.get_required_role_by_ap(ap)  # 'drones'|'gvs'|'humans'|'unknown'

    # ------------------------ candidate construction --------------------------
    def _grouped_candidates(
        self, unlocked: Set[str], completed: Set[str], aps_now: Set[str]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        {role: [(ap, group_key), ...]} filtered as:
         - unlocked & not completed & not currently true
         - not environment APs
         - HUMANS sticky (if a human already bound to group, skip); drones/GVs not sticky
         - within a group, only the NEXT unfinished task (step-by-step)
        """
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

            # STICKINESS ONLY FOR HUMANS:
            if role == "humans":
                try:
                    if bm.get_bound_agent_for_group(group, agent_type="humans") is not None:
                        # a human is already working this group; keep it with them
                        continue
                except Exception:
                    pass
            # NOTE: drones/gvs are intentionally NOT skipped if already bound:
            # they may take the next physical AP in the same group.

            raw[role].append((ap, group))

        # Reduce to the next unfinished AP per group
        per_role: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for role, items in raw.items():
            by_group: Dict[str, List[str]] = defaultdict(list)
            for ap, g in items:
                by_group[g].append(ap)
            for g, aps in by_group.items():
                ordered = lbl.get_group_ordered_tasks(g) or []
                nxt = None
                for t in ordered:
                    if (t in aps) and (t not in completed):
                        nxt = t
                        break
                if nxt is not None:
                    per_role[role].append((nxt, g))
        return per_role

    # ------------------------------- API --------------------------------------
    def choose_eta(
        self, unlocked: Set[str], completed: List[str], aps: Set[str]
    ) -> Dict[Agent, str]:
        completed_set = set(completed)
        actions: Dict[Agent, str] = {}

        cand = self._grouped_candidates(unlocked, completed_set, aps)

        # Global guard: at most one AP per group this tick (across roles)
        claimed_groups: Set[str] = set()

        def greedy_pick_for_role(role: str) -> None:
            pool: List[Tuple[str, str]] = [
                (ap, g) for (ap, g) in cand.get(role, []) if g not in claimed_groups
            ]
            if not pool:
                return

            free_agents: List[Agent] = [
                a for a in self.agents_by_type.get(role, []) if a not in actions
            ]
            if not free_agents:
                return

            while free_agents and pool:
                # precompute features for pool
                feats: List[Tuple[str, int, float, str]] = []  # (ap, tid, v, group)
                for ap, g in pool:
                    tid = self._parse_tid(ap)
                    q = self.labeler.states.get(g)
                    s = build_s_vector(self.ws, ap)
                    v = float(self.value_bank.value_leaf(ap, q, s))
                    feats.append((ap, tid, v, g))

                best: Optional[Tuple[float, int, int]] = None  # (score, ai, pi)
                for ai, agent in enumerate(free_agents):
                    for pi, (ap, tid, v, _g) in enumerate(feats):
                        eta = self._eta_pos(agent, tid, use_goal=True)
                        score = v - self.eta_weight * eta
                        if (best is None) or (score > best[0]):
                            best = (score, ai, pi)

                if best is None:
                    break

                _, ai, pi = best
                agent = free_agents.pop(ai)
                ap, _tid, _v, g = feats[pi]

                # bind (task_name, agent_obj, agent_type_str)
                ok = False
                try:
                    ok = self.binding_manager.record_assignment(ap, agent, role)
                except TypeError:
                    ok = self.binding_manager.record_assignment(
                        task_name=ap, agent=agent, agent_type=role
                    )
                if ok:
                    actions[agent] = ap
                    claimed_groups.add(g)
                    pool = [(t, gg) for (t, gg) in pool if gg not in claimed_groups]
                # else: rejected; try other options

        greedy_pick_for_role("drones")
        greedy_pick_for_role("gvs")
        greedy_pick_for_role("humans")
        return actions
