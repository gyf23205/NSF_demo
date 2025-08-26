from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import math
import numpy as np

# Core codebase imports
from ltl_core.agent import Agent
from ltl_core.specification import get_ap_prefix, is_environment_ap, Specification
from ltl_core.binding_manager import BindingManager
from ltl_core.labeler import Labeler
from ltl_core.workspace import Workspace
from ltl_core.value_fn import ValueBank

# Feature builder you already use for training
from rl.value_features import build_s_vector


class RLAllocator:
    """
    Value-aware function allocator.

    Big ideas:
    - Build the *same* candidate set the dev allocator uses:
        • start from Labeler.unlocked set
        • drop environment APs
        • group-aware, one agent per (group, role)
        • obey pairwise ordering within each group (nav→scan, ...)

    - Score feasible (agent, ap) pairs as:
        score = V(q_group, s(ap)) - w_eta * ETA(agent, tid) - w_dv * ΔV(agent)

    - Record bindings at assignment time using BindingManager, so subsequent
      completion checks (and releases at accepting DFA states) work as before.
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

        # convenience: cached lists (kept in sync by caller when env is (re)built)
        self._drones = list(self.agents_by_type.get("drones", []))
        self._gvs = list(self.agents_by_type.get("gvs", []))
        self._humans = list(self.agents_by_type.get("humans", []))

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _parse_tid(ap: str) -> int:
        """
        Extract the target id from an AP name, e.g. p_nav_7_1_1_0 -> 7.
        Returns -1 if not present.
        """
        try:
            return int(ap.split("_")[2])
        except Exception:
            return -1

    def _eta_pos(self, agent: Agent, tid: int, use_goal: bool = True) -> float:
        """
        ETA proxy in seconds: Euclidean distance / max_speed
        (matches what you used during training; simple and monotonic).
        """
        if tid < 0 or tid >= len(getattr(self.ws, "target_locations", [])):
            return 1e9

        try:
            if use_goal and getattr(agent, "goal", None) is not None:
                ax, ay = float(agent.goal[0]), float(agent.goal[1])
            else:
                ax, ay = float(agent.pos[0]), float(agent.pos[1])

            tx, ty = self.ws.target_locations[int(tid)]
            tx, ty = float(tx), float(ty)

            dist = math.hypot(tx - ax, ty - ay)
            vmax = float(getattr(agent, "max_speed", 1.0)) or 1.0
            return dist / max(vmax, 1e-6)
        except Exception:
            return 1e9

    def _delta_v(self, agent: Agent) -> float:
        """
        Optional ΔV term for regularization; default 0 to match current training.
        Override if you later log per-agent speed/effort.
        """
        return 0.0

    def _role_for_ap(self, ap: str) -> str:
        """Map an AP to 'drones' / 'gvs' / 'humans' (or 'unknown')."""
        return self.spec.get_required_role_by_ap(ap)

    # --------------------------------------------------------------------- #
    # Candidate construction (mirrors RandomAllocator semantics)
    # --------------------------------------------------------------------- #

    def _grouped_candidates(
        self, unlocked: Set[str], completed: Set[str], aps_now: Set[str]
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Return {role: [(ap, group_key), ...]} filtered as:
          - in unlocked, not completed, not already true now (aps_now)
          - not environment APs
          - group permits this role (BindingManager.group_types)
          - one agent per (group, role): skip if that (group,role) is currently bound
          - within a group, only offer the *next* unfinished task according to
            labeler.get_group_ordered_tasks(...)
        """
        bm = self.binding_manager
        lbl = self.labeler

        # Base raw candidates per role
        raw: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for ap in unlocked:
            if ap in completed or ap in aps_now:
                continue
            if is_environment_ap(ap):
                continue

            role = self._role_for_ap(ap)
            if role == "unknown":
                continue

            group = bm.task_to_group.get(ap)
            if not group:
                # leaf AP that isn't part of a registered group (rare)
                continue

            # enforce at most one agent per (group, role)
            if bm.get_bound_agent_for_group(group, agent_type=role):
                continue

            # check role allowed for this group (extra guard)
            allow = bm.group_types.get(group, set())
            if allow and (role not in allow):
                continue

            raw[role].append((ap, group))

        # Reduce to the *next unfinished* AP within each group
        per_role: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        for role, items in raw.items():
            # bucket by group
            by_group: Dict[str, List[str]] = defaultdict(list)
            for ap, g in items:
                by_group[g].append(ap)

            for g, aps in by_group.items():
                ordered = lbl.get_group_ordered_tasks(g) or []
                # choose first task in order that is in *aps* and not completed
                next_task: Optional[str] = None
                for t in ordered:
                    if (t in aps) and (t not in completed):
                        next_task = t
                        break
                if next_task is not None:
                    per_role[role].append((next_task, g))

        return per_role

    # --------------------------------------------------------------------- #
    # Public API used by Simulation.step(...)
    # --------------------------------------------------------------------- #

    def choose_eta(
        self, unlocked: Set[str], completed: List[str], aps: Set[str]
    ) -> Dict[Agent, str]:
        """
        Value-aware selection. Returns a (possibly empty) mapping {Agent: ap}.
        Called after completions / priority changes.
        """
        completed_set = set(completed)
        actions: Dict[Agent, str] = {}
        assigned: Set[Agent] = set()

        # 1) Build candidates with the same semantics as the dev allocator
        cand = self._grouped_candidates(unlocked, completed_set, aps)

        # 2) Pick per-role using learned value minus ETA/dV penalties
        def pick_for_role(role: str) -> None:
            items = cand.get(role, [])
            if not items:
                return

            free_agents: List[Agent] = [
                a for a in self.agents_by_type.get(role, []) if a not in assigned
            ]
            if not free_agents:
                return

            best: Optional[Tuple[float, Agent, str]] = None

            for agent in free_agents:
                for ap, group in items:
                    tid = self._parse_tid(ap)
                    # DFA node for this group's current state
                    q_node = self.labeler.states.get(group)
                    # s-vector must match training features
                    s_vec = build_s_vector(self.ws, ap)

                    # learned value
                    v = float(self.value_bank.value_leaf(ap, q_node, s_vec))

                    # simple ETA (goal-aware to reduce oscillation)
                    eta = self._eta_pos(agent, tid, use_goal=True)

                    # optional ΔV
                    dv = self._delta_v(agent)

                    score = v - self.eta_weight * eta - self.dv_weight * dv
                    if (best is None) or (score > best[0]):
                        best = (score, agent, ap)

            if best is None:
                return

            _, agent, ap = best

            # Respect/record binding (required for Labeler to validate completions)
            role_name = role  # already 'drones'/'gvs'/'humans'
            ok = self.binding_manager.record_assignment(ap, agent, role_name)
            if ok:
                actions[agent] = ap
                assigned.add(agent)
            # else: somebody else already bound the group since we built candidates

        pick_for_role("drones")
        pick_for_role("gvs")
        pick_for_role("humans")

        return actions
