from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable

import numpy as np
from rl.buffer import APReplay, APTransition
from ltl_core.value_fn import ValueBank

@dataclass
class TrainerCfg:
    batch_size: int = 64          # samples per update() call
    alpha: float = 1e-3           # TD step size (overrides ValueBank LR if set)
    updates_per_call: int = 1     # do multiple minibatches per call
    bootstrap_next: bool = True   # use V(q_after, s) as bootstrap (TD(0))
    clip_td: Optional[float] = None  # clip TD magnitude for stability

class CriticTrainer:
    """
    TD(0) trainer for local value functions V(q, s) shared across regions.
    Assumes 'tau' already includes any penalties (e.g., oversight).
    """

    def __init__(self, V: ValueBank, replay: APReplay, cfg: TrainerCfg = TrainerCfg()):
        self.V = V
        self.buf = replay
        self.cfg = cfg
        self._td2_ma = 0.0  # moving average of TD^2 for quick health check

    # ---- Online path: call this once per AP completion -------------------

    def push_completion(self, tr: APTransition | None = None, *,
                        ap: str = "", q_before=None, q_after=None, s_vec=None, tau: float = 0.0, meta=None):
        """
        Either pass a ready-made APTransition via 'tr' OR individual fields via keywords.
        """
        if tr is not None:
            self.buf.push(tr.ap, tr.q_before, tr.q_after, tr.s_vec, tr.tau, tr.meta)
        else:
            self.buf.push(ap, q_before, q_after, s_vec, tau, meta)

    def update_online(self, n_recent: int = 1):
        """
        Do up to n_recent most-recent online TD updates (no sampling).
        Useful when you want immediate learning after each completion.
        """
        recent = self.buf.recent(n_recent)
        for tr in recent:
            self._td_step(tr)

    # ---- Batch path: call this periodically (e.g., every few steps) -----

    def update(self):
        """
        Sample a mini-batch from replay and apply TD(0) updates.
        """
        if len(self.buf) == 0:
            return {"size": 0, "td2_ma": self._td2_ma}

        for _ in range(self.cfg.updates_per_call):
            batch = self.buf.sample(self.cfg.batch_size)
            for tr in batch:
                self._td_step(tr)

        return {"size": len(self.buf), "td2_ma": self._td2_ma}

    # ---- Core TD step ----------------------------------------------------

    def _td_step(self, tr: APTransition):
        v  = self.V.value_leaf(tr.ap, tr.q_before, tr.s_vec)
        if self.cfg.bootstrap_next:
            vp = self.V.value_leaf(tr.ap, tr.q_after, tr.s_vec)
        else:
            vp = 0.0

        td = tr.tau + vp - v                       # TD(0)
        if self.cfg.clip_td is not None:
            td = float(max(-self.cfg.clip_td, min(self.cfg.clip_td, td)))

        # Semi-gradient step on V
        self.V.backprop_leaf(tr.ap, tr.q_before, tr.s_vec, td, alpha=self.cfg.alpha)

        # Diagnostics: TD^2 moving average
        self._td2_ma = 0.98 * self._td2_ma + 0.02 * float(td * td)

