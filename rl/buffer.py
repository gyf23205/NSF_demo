from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque
import random
import pickle
import time
import numpy as np

@dataclass
class APTransition:
    """
    One sample = one completed AP execution.
    ap:        leaf AP name (e.g., 'p_nav_3_1_1_0')
    q_before:  DFA node before execution (any object; used via ValueBank's indexer)
    q_after:   DFA node after completion
    s_vec:     np.ndarray (s_dim,) - region/layout features captured at dispatch time
    tau:       float - elapsed time for this AP (+ penalties already applied)
    meta:      dict - optional: episode, step indices, agent id, target id, timestamps
    ts:        float - insertion time (sec) for debugging/bookkeeping
    """
    ap: str
    q_before: Any
    q_after: Any
    s_vec: np.ndarray
    tau: float
    meta: Dict[str, Any]
    ts: float

class APReplay:
    """Simple FIFO replay for AP-level transitions."""
    def __init__(self, capacity: int = 100_000, seed: Optional[int] = None):
        self.capacity = int(capacity)
        self.mem: Deque[APTransition] = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)

    # ----------------- Write API -----------------

    def push(
        self,
        ap: str,
        q_before: Any,
        q_after: Any,
        s_vec: np.ndarray,
        tau: float,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mem.append(APTransition(
            ap=ap,
            q_before=q_before,
            q_after=q_after,
            s_vec=np.asarray(s_vec, dtype=float),
            tau=float(tau),
            meta=dict(meta or {}),
            ts=time.time(),
        ))

    def push_many(self, items: Iterable[Tuple[str, Any, Any, np.ndarray, float, Optional[Dict[str, Any]]]]):
        for ap, q_b, q_a, s, tau, meta in items:
            self.push(ap, q_b, q_a, s, tau, meta)

    # ----------------- Read API ------------------

    def sample(self, n: int) -> List[APTransition]:
        """Uniform sample without replacement; returns <= n if buffer smaller."""
        n = min(n, len(self.mem))
        if n <= 0:
            return []
        # Convert to list once to avoid O(n) pops on deque for random sample
        return self.rng.sample(list(self.mem), n)

    def recent(self, n: int) -> List[APTransition]:
        """Return the most recent n items (no removal)."""
        n = min(n, len(self.mem))
        if n <= 0:
            return []
        # Right end is the newest
        return list(self.mem)[-n:]

    def __len__(self) -> int:
        return len(self.mem)

    def clear(self) -> None:
        self.mem.clear()

    # ----------------- Persistence ----------------

    def save(self, path: str) -> None:
        """Pickle the whole buffer (handles arbitrary q-node objects)."""
        with open(path, "wb") as f:
            pickle.dump({
                "capacity": self.capacity,
                "mem": list(self.mem),
                "rng_state": self.rng.getstate(),
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.capacity = int(obj.get("capacity", self.capacity))
        self.mem = deque(obj.get("mem", []), maxlen=self.capacity)
        rng_state = obj.get("rng_state", None)
        if rng_state is not None:
            self.rng.setstate(rng_state)

    # ----------------- Small utilities ------------

    def stats(self) -> Dict[str, Any]:
        """Quick diagnostics."""
        size = len(self.mem)
        taus = [t.tau for t in self.mem[-min(2048, size):]]  # sample recent
        return {
            "size": size,
            "tau_mean_recent": float(np.mean(taus)) if taus else None,
            "tau_std_recent": float(np.std(taus)) if taus else None,
            "unique_aps_recent": len(set(t.ap for t in self.mem[-min(2048, size):])),
        }
