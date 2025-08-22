from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional

import math
import re
import torch
import torch.nn as nn
import torch.optim as optim

from ltl_core.specification import get_ap_prefix  # keeps grouping consistent with codebase


# ---------- Configs ----------

@dataclass
class ValueNetConfig:
    s_dim: int                  # dimension of s vector passed into the net
    state_emb_dim: int = 16     # embedding size for DFA state id
    hidden_dim: int = 64        # width of MLP
    lr: float = 1e-3            # learning rate per template net
    max_states: int = 4096      # maximum distinct DFA states we expect per template
    device: str = "cpu"         # "cuda" if available
    dtype: torch.dtype = torch.float32


# ---------- Utilities ----------

class StateIndexer:
    """
    Assigns stable, consecutive integer ids to (template, DFA state) pairs on demand.
    It does not assume any particular type for q_node; we hash by str(q_node) unless it has 'name'/'id'.
    """
    def __init__(self):
        # Per-template: map from state_key -> int
        self._index: Dict[str, Dict[str, int]] = {}
        # Per-template: next available id (starting from 0)
        self._next_id: Dict[str, int] = {}

    @staticmethod
    def _key_for_q(q_node: Any) -> str:
        # Try common fields first; fall back to repr
        for attr in ("id", "name", "label"):
            if hasattr(q_node, attr):
                return f"{attr}:{getattr(q_node, attr)}"
        return repr(q_node)

    def get_id(self, template: str, q_node: Any) -> int:
        if template not in self._index:
            self._index[template] = {}
            self._next_id[template] = 0
        table = self._index[template]
        key = self._key_for_q(q_node)
        if key not in table:
            table[key] = self._next_id[template]
            self._next_id[template] += 1
        return table[key]

    def size(self, template: str) -> int:
        return self._next_id.get(template, 0)


# ---------- Models ----------

class LocalValueNet(nn.Module):
    """
    Small MLP: V(q, s) with an embedding lookup for q (DFA state id)
    and a vector encoder for s (region/layout features).
    """
    def __init__(self, cfg: ValueNetConfig):
        super().__init__()
        self.cfg = cfg
        self.state_emb = nn.Embedding(cfg.max_states, cfg.state_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.state_emb_dim + cfg.s_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

        # Kaiming init for MLP
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, q_ids: torch.Tensor, s_vec: torch.Tensor) -> torch.Tensor:
        """
        q_ids: (B,) long tensor of DFA state ids
        s_vec: (B, s_dim) float tensor
        returns: (B, 1) values
        """
        q_emb = self.state_emb(q_ids)               # (B, state_emb_dim)
        x = torch.cat([q_emb, s_vec], dim=-1)       # (B, state_emb_dim + s_dim)
        v = self.mlp(x)                              # (B, 1)
        return v


# ---------- ValueBank (public API) ----------

class ValueBank:
    """
    Manages one LocalValueNet per function template, with its optimizer.
    Template is determined by get_ap_prefix(ap_name), so all regions that share logic share weights.
    """
    def __init__(self, cfg: ValueNetConfig):
        self.cfg = cfg
        self.state_indexer = StateIndexer()
        self._nets: Dict[str, LocalValueNet] = {}
        self._opts: Dict[str, optim.Optimizer] = {}
        self.device = torch.device(cfg.device)

    # --- Core helpers ---

    def _template_key(self, ap_name: str) -> str:
        # Use canonical prefix so grouping is consistent with the rest of the codebase.
        return get_ap_prefix(ap_name)

    def _ensure_net(self, template: str):
        if template not in self._nets:
            net = LocalValueNet(self.cfg).to(self.device)
            opt = optim.Adam(net.parameters(), lr=self.cfg.lr)
            self._nets[template] = net
            self._opts[template] = opt

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=self.cfg.dtype)
        return torch.tensor(x, device=self.device, dtype=self.cfg.dtype)

    # --- Public API ---

    @torch.no_grad()
    def value_leaf(self, ap_name: str, q_node: Any, s_vec) -> float:
        """
        Returns scalar V(q, s) as a Python float for a leaf-level formula 'ap_name'.
        """
        template = self._template_key(ap_name)
        self._ensure_net(template)

        q_id = self.state_indexer.get_id(template, q_node)
        q_ids = torch.tensor([q_id], device=self.device, dtype=torch.long)
        s = self._to_tensor(s_vec).reshape(1, -1)

        v = self._nets[template](q_ids, s)  # (1,1)
        return float(v.item())

    def value_leaf_batch(self, ap_names, q_nodes, s_mat) -> torch.Tensor:
        """
        Batched value for efficiency: all tensors returned (shape: (B,1)).
        ap_names: list[str] length B
        q_nodes:  list[Any] length B
        s_mat:    (B, s_dim) array-like
        """
        assert len(ap_names) == len(q_nodes), "ap_names and q_nodes must have same length"
        B = len(ap_names)
        # Group by template to minimize embedding lookups
        idx_by_template: Dict[str, list] = {}
        for i, ap in enumerate(ap_names):
            t = self._template_key(ap)
            idx_by_template.setdefault(t, []).append(i)

        out = torch.zeros((B, 1), device=self.device, dtype=self.cfg.dtype)
        s_all = self._to_tensor(s_mat).reshape(B, -1)

        for t, idxs in idx_by_template.items():
            self._ensure_net(t)
            net = self._nets[t]

            q_ids_list = [self.state_indexer.get_id(t, q_nodes[i]) for i in idxs]
            q_ids = torch.tensor(q_ids_list, device=self.device, dtype=torch.long)
            s_chunk = s_all[idxs, :]
            v_chunk = net(q_ids, s_chunk)  # (len(idxs), 1)
            out[idxs, :] = v_chunk

        return out  # (B,1)

    def backprop_leaf(self, ap_name: str, q_node: Any, s_vec, td_error: float, alpha: Optional[float] = None):
        """
        Applies a simple semi-gradient TD(0) step:
            Loss = 0.5 * (TD)^2, where TD is provided by the caller.
        If 'alpha' is provided, it temporarily scales the optimizer LR for this step.
        """
        template = self._template_key(ap_name)
        self._ensure_net(template)

        net = self._nets[template]
        opt = self._opts[template]

        q_id = self.state_indexer.get_id(template, q_node)
        q_ids = torch.tensor([q_id], device=self.device, dtype=torch.long)
        s = self._to_tensor(s_vec).reshape(1, -1)

        # Optional per-call LR scaling (keeps a stable base LR but allows a TD-specific step size)
        if alpha is not None:
            base_lrs = [pg['lr'] for pg in opt.param_groups]
            for pg in opt.param_groups:
                pg['lr'] = alpha

        net.train()
        opt.zero_grad(set_to_none=True)
        v = net(q_ids, s)                                    # (1,1)
        td_t = torch.tensor([[td_error]], device=self.device, dtype=self.cfg.dtype)
        target = (v.detach() + td_t)                         # target = τ + V(next), no grad through target
        loss = 0.5 * (v - target).pow(2).mean()              # depends on v => gradients flow
        loss.backward()
        opt.step()

        if alpha is not None:
            # Restore base LRs
            for pg, lr0 in zip(opt.param_groups, base_lrs):
                pg['lr'] = lr0

    # --- Persistence ---

    def save(self, path: str):
        """
        Saves all template nets + optimizers + indexer state.
        """
        state = {
            "cfg": self.cfg.__dict__,
            "nets": {t: net.state_dict() for t, net in self._nets.items()},
            "opts": {t: opt.state_dict() for t, opt in self._opts.items()},
            "indexer": {
                "index": self.state_indexer._index,
                "next_id": self.state_indexer._next_id,
            },
        }
        torch.save(state, path)

    def load(self, path: str, map_location: Optional[str] = None):
        """
        Loads template nets + optimizers + indexer state. Call before using the bank.
        """
        loc = map_location or self.device.type
        state = torch.load(path, map_location=loc)

        # Restore cfg (allow using existing cfg.s_dim/device if different)
        cfg_dict = state.get("cfg", {})
        # Keep current s_dim/device if user changed them in code; otherwise use saved.
        merged = {**cfg_dict, "s_dim": self.cfg.s_dim, "device": self.cfg.device}
        self.cfg = ValueNetConfig(**merged)

        # Restore indexer
        self.state_indexer._index = state["indexer"]["index"]
        self.state_indexer._next_id = state["indexer"]["next_id"]

        # Rebuild nets/opts and load weights
        self._nets.clear()
        self._opts.clear()
        for t, net_sd in state["nets"].items():
            self._ensure_net(t)
            self._nets[t].load_state_dict(net_sd)
        for t, opt_sd in state["opts"].items():
            self._opts[t].load_state_dict(opt_sd)

    # --- Convenience ---

    def to(self, device: str):
        """
        Move all template nets to a new device (e.g., 'cuda').
        """
        self.device = torch.device(device)
        for net in self._nets.values():
            net.to(self.device)

    def templates(self):
        return list(self._nets.keys())
    