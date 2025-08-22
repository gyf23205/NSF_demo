from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

@dataclass
class AgentView:
    idx: int           # contiguous index for masks/scores
    obj: object        # actual Agent object (ltl_core.agent.Agent)
    role: str          # "drones" | "gvs" | "humans"
    busy: bool = False # already assigned this tick?

@dataclass
class TaskView:
    idx: int           # contiguous index for masks/scores
    name: str          # AP name (leaf)
    template: str      # get_ap_prefix(AP)
    group: str         # group id this AP belongs to (binding group)
    target_id: Optional[int] = None

@dataclass
class Obs:
    agents: List[AgentView]
    tasks: List[TaskView]
    # Feasibility mask: 1 if (agent, task) is allowed (role, binding, group limits), else 0.
    mask: List[List[int]]       # shape = [len(agents)][len(tasks)]
    # Score matrix: lower is better (e.g., ETA + ΔV). Use a large number for invalid pairs.
    score: List[List[float]]    # same shape as mask

class MaskedMatchPolicy:
    """
    Masked greedy matcher: assigns at most one task per agent and one agent per task.
    It never calls binding manager; it assumes 'mask' already encodes feasibility.
    """

    def __init__(self, tie_break_margin: float = 1e-6):
        self.tie_break_margin = tie_break_margin

    def act(self, obs: Obs) -> Dict[object, str]:
        A = obs.agents
        T = obs.tasks
        M = obs.mask
        S = obs.score

        if not A or not T:
            return {}

        # Build a flat list of feasible (score, a_idx, t_idx)
        pairs = []
        for ai, a in enumerate(A):
            if a.busy:
                continue
            row_m = M[ai]
            row_s = S[ai]
            for ti, t in enumerate(T):
                if row_m[ti] == 1:
                    pairs.append((row_s[ti], ai, ti))

        # Nothing feasible
        if not pairs:
            return {}

        # Sort by score (asc). Tie-break by ai, then ti for determinism.
        pairs.sort(key=lambda x: (x[0], x[1], x[2]))

        assigned_agents = set()
        claimed_tasks = set()
        out: Dict[object, str] = {}

        for score, ai, ti in pairs:
            if ai in assigned_agents or ti in claimed_tasks:
                continue
            a = A[ai]
            t = T[ti]
            out[a.obj] = t.name
            assigned_agents.add(ai)
            claimed_tasks.add(ti)

        return out
