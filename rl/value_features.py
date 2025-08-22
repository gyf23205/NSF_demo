from __future__ import annotations
import re
import numpy as np
from typing import Tuple

_TID_RE = re.compile(r".*?_(\d+)_")

def _parse_tid(ap: str) -> int:
    m = _TID_RE.match(ap)
    return int(m.group(1)) if m else -1

def _norm_xy(rc: Tuple[int,int], grid_size: Tuple[int,int]) -> Tuple[float,float]:
    r, c = rc
    H, W = grid_size
    # center-of-cell, normalized to [0,1]
    x = (c + 0.5) / max(W, 1)
    y = (r + 0.5) / max(H, 1)
    return float(x), float(y)

def build_s_vector(ws, ap: str) -> np.ndarray:
    """
    Minimal s: [x_norm, y_norm, priority_norm].
    - Uses workspace.target_locations[tid] (row,col) and workspace.grid_size.
    - priority_norm = priority / max_priority (assume {1,2} -> /2.0 by default).
    """
    tid = _parse_tid(ap)
    H, W = getattr(ws, "grid_size", (1, 1))
    targets = getattr(ws, "target_locations", [])
    prio_fn = getattr(ws, "get_target_priority", None)

    # coords
    if 0 <= tid < len(targets):
        x, y = _norm_xy(targets[tid], (H, W))
    else:
        x, y = 0.0, 0.0

    # priority
    if callable(prio_fn):
        pr = float(prio_fn(tid))
        pr_norm = pr / 2.0  # adjust if your max priority differs
    else:
        pr_norm = 0.0

    return np.array([x, y, pr_norm], dtype=np.float32)
