from __future__ import annotations
from typing import Dict, Set, List, Tuple
import numpy as np
from ltl_core.allocator import RandomAllocator
from ltl_core.specification import get_ap_prefix, AP_TYPE_PREFIX_MAP
from ltl_core.value_fn import ValueBank
from ltl_core.agent import Agent


class RLAllocator(RandomAllocator):
    """
    Greedy allocator that combines an ETA proxy with ΔV = V(q', s) - V(q, s).
    Lower score is better. Respects group/type bindings via BindingManager.
    """

    def __init__(
        self,
        spec,
        agents_by_type,
        binding_manager,
        labeler,
        workspace=None,
        value_bank: ValueBank | None = None,
        eta_weight: float = 1.0,
        dv_weight: float = 1.0,
    ):
        super().__init__(spec, agents_by_type, binding_manager, labeler, workspace)
        self.value_bank = value_bank
        self.eta_weight = float(eta_weight)
        self.dv_weight = float(dv_weight)

    # Simulator calls this entry point each replan tick.
    # Keep the signature to match Simulation.step(). :contentReference[oaicite:5]{index=5}
    def choose_eta(self, unlocked: Set[str], completed: List[str], aps: Set[str]) -> Dict[Agent, str]:
        actions: Dict[Agent, str] = {}
        assigned = set()

        # 0) keep valid in-progress human symbolic tasks (respect binding) :contentReference[oaicite:6]{index=6}
        for agent in self.agents_by_type.get("humans", []):
            task = getattr(agent, "current_symbolic_task", None)
            if task and (task in unlocked) and (task not in completed) and (task not in aps):
                agent_type = self.spec.get_required_role_by_ap(task)
                if self.binding_manager.record_assignment(task, agent, agent_type):
                    actions[agent] = task
                    assigned.add(agent)
                else:
                    if hasattr(agent, "reset_symbolic"):
                        agent.reset_symbolic()

        # workspace layout mask s (used by V(q,s))
        s = np.asarray(getattr(self.workspace, "target_mask", []), dtype=np.float32)

        # 1) build and score (agent, task) candidates across groups
        scored: List[Tuple[float, object, str]] = []  # (score, agent, ap)

        for group, _tasks in self.binding_manager.group_to_tasks.items():
            ordered = self.labeler.get_group_ordered_tasks(group) or []
            q_now = self.labeler.states.get(group, None)
            dfa = self.binding_manager.group_to_automaton.get(group, None)

            for t in ordered:
                if (t not in unlocked) or (t in completed) or (t in aps):
                    continue

                role = self.spec.get_required_role_by_ap(t)
                if role not in self.agents_by_type:
                    continue

                pref = get_ap_prefix(t)

                # compute team-induced q' for this group's DFA under AP t :contentReference[oaicite:7]{index=7}
                q_next = None
                if (dfa is not None) and (q_now is not None):
                    try:
                        q_next = self.labeler._step_dfa(dfa, q_now, {t})
                    except Exception:
                        q_next = q_now

                # ΔV term from learned local value; safe if bank not ready yet
                dV = 0.0
                if self.value_bank is not None and (q_now is not None) and (q_next is not None):
                    try:
                        v_now = self.value_bank.predict(group, q_now, s)
                        v_nxt = self.value_bank.predict(group, q_next, s)
                        dV = float(v_nxt - v_now)
                    except Exception:
                        dV = 0.0

                for a in self.agents_by_type.get(role, []):
                    if a in assigned:
                        continue
                    # respect binding; skip if this (group,role) is bound to someone else :contentReference[oaicite:8]{index=8}
                    if not self.binding_manager.record_assignment(t, a, role):
                        continue

                    eta = self._eta_estimate(a, t, pref)
                    score = self.eta_weight * eta + self.dv_weight * dV

                    # tiny tie-break to prefer keeping an existing bound agent, if any
                    bound = self.binding_manager.get_bound_agent_for_group(group, role)
                    if bound is not None and getattr(bound, "label", None) == getattr(a, "label", None):
                        score -= 0.1

                    scored.append((score, a, t))

        # 2) greedy pick best unique agent-task pairs
        scored.sort(key=lambda x: x[0])
        claimed: Set[str] = set()
        for _, agent, ap in scored:
            if agent in assigned or ap in claimed:
                continue
            role = self.spec.get_required_role_by_ap(ap)
            if not self.binding_manager.record_assignment(ap, agent, role):
                continue
            actions[agent] = ap
            assigned.add(agent)
            claimed.add(ap)

        return actions

    # ---- helpers ---------------------------------------------------------
    def _eta_estimate(self, agent, ap: str, pref: str) -> float:
        """Simple ETA proxy: symbolic uses remaining progress; physical uses distance/speed."""
        ap_type = AP_TYPE_PREFIX_MAP.get(pref)
        if ap_type == "symbolic":
            try:
                remaining = max(0.0, 1.0 - float(agent.get_progress(ap)))
            except Exception:
                remaining = 1.0
            return 10.0 * remaining  # nominal symbolic duration

        # physical: euclidean distance to target/dropoff in grid cells / speed :contentReference[oaicite:9]{index=9} :contentReference[oaicite:10]{index=10}
        try:
            tid = int(ap.split("_")[2])
        except Exception:
            tid = 0

        if pref == "p_dropoff":
            goal = getattr(self.workspace, "dropoff_locations", {}).get(tid)
        else:
            locs = getattr(self.workspace, "target_locations", [])
            goal = locs[tid] if 0 <= tid < len(locs) else None

        if goal is None:
            return 1e6  # impossible → effectively filtered out

        dx = float(agent.pos[0]) - float(goal[0])
        dy = float(agent.pos[1]) - float(goal[1])
        dist = (dx * dx + dy * dy) ** 0.5
        speed = float(getattr(agent, "speed", 1.0)) or 1.0
        return dist / speed
