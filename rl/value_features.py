from __future__ import annotations
from typing import List, Tuple
import numpy as np

# Import only light dependencies from your codebase
from ltl_core.specification import get_ap_prefix
from ltl_core.workspace import Workspace

def _parse_tid(ap: str) -> int:
    try:
        return int(ap.split("_")[2])
    except Exception:
        return -1

def _role_for_prefix(prefix: str) -> str:
    """
    Map function template to agent role, consistent with your spec/bindings.
    Adjust if your mapping differs (e.g., verify -> humans).
    """
    if prefix.startswith("p_nav") or prefix.startswith("p_scan"):
        return "drones"
    if prefix.startswith("p_pickup") or prefix.startswith("p_dropoff"):
        return "gvs"
    if prefix.startswith("p_verify") or prefix.startswith("p_priority") or prefix.startswith("p_triage") or prefix.startswith("p_atmconfirm"):
        return "humans"
    # default: unknown role (we’ll still produce features, but nearest-agent distance will be 1.0)
    return "unknown"

def _nearest_agent_dist(ws: Workspace, role: str, target_xy: Tuple[float, float]) -> float:
    if role not in ws.agents:
        return 1.0
    agents = ws.agents.get(role, [])
    if not agents:
        return 1.0
    tx, ty = float(target_xy[0]), float(target_xy[1])
    best = None
    for a in agents:
        ax, ay = float(a.pos[0]), float(a.pos[1])
        d = np.hypot(tx - ax, ty - ay)
        best = d if best is None else min(best, d)
    return float(best if best is not None else 1.0)

def _map_diagonal(ws: Workspace) -> float:
    h, w = ws.size
    return float(np.hypot(h, w))

def _remaining_work_density(ws: Workspace) -> float:
    """
    A light proxy for 'how much is left':
    fraction of target regions that are active (mask==1) and not yet fully resolved.
    If you later track per-target resolution flags, plug them here.
    """
    mask = getattr(ws, "target_mask", None)
    if mask is None or len(mask) == 0:
        return 0.0
    active = sum(1 for m in mask if m)
    # If you maintain resolved flags, you can subtract them here.
    # For now, just normalize active by total.
    return float(active) / float(len(mask))

def build_s_vector(ws: Workspace, ap: str, s_dim: int = 3) -> List[float]:
    """
    Build a compact feature vector used by the critic:
      [ priority_norm, dist_norm, remaining_density ]

    - priority_norm: target priority in [0,2] mapped to [0,1]
    - dist_norm: nearest feasible agent distance to target, divided by map diagonal
    - remaining_density: fraction of (currently) active regions

    If your model was created with s_dim==3, keep it. If you later expand features,
    you can append additional terms here (be sure to retrain).
    """
    # --- priority feature ---
    tid = _parse_tid(ap)
    prio = 0
    if tid >= 0 and hasattr(ws, "get_target_priority"):
        try:
            prio = int(ws.get_target_priority(tid))
        except Exception:
            prio = 0
    priority_norm = float(np.clip(prio, 0, 2)) / 2.0  # ∈ [0,1]

    # --- distance-to-target (nearest feasible agent for this AP) ---
    target_xy = (0.0, 0.0)
    if tid >= 0 and hasattr(ws, "target_locations"):
        locs = ws.target_locations
        if tid < len(locs):
            target_xy = tuple(locs[tid])
    prefix = get_ap_prefix(ap)
    role = _role_for_prefix(prefix)
    dist = _nearest_agent_dist(ws, role, target_xy)
    diag = max(_map_diagonal(ws), 1e-6)
    dist_norm = float(np.clip(dist / diag, 0.0, 1.0))

    # --- remaining work density ---
    remaining_density = float(np.clip(_remaining_work_density(ws), 0.0, 1.0))

    s = [priority_norm, dist_norm, remaining_density]

    # Pad or trim to s_dim for robustness
    if len(s) < s_dim:
        s = s + [0.0] * (s_dim - len(s))
    elif len(s) > s_dim:
        s = s[:s_dim]
    return s
