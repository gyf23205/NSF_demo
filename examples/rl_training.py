# Train & test a value-aware function allocator using your existing simulator.

from __future__ import annotations
import argparse
import os
from collections import defaultdict

import sys
from pathlib import Path
my_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(my_path)

import numpy as np

# --- Your codebase imports ---
from ltl_core.specification import Specification
from ltl_core.labeler import Labeler
from ltl_core.workspace import Workspace
from ltl_core.simulation import Simulation
from ltl_core.binding_manager import BindingManager

# Value function + RL glue
from ltl_core.value_fn import ValueNetConfig, ValueBank
from rl.value_features import build_s_vector
from rl.buffer import APReplay
from rl.trainer import CriticTrainer, TrainerCfg
from rl.policy import MaskedMatchPolicy
from rl.allocator_rl import RLAllocator


# -------------------- small helpers --------------------

def make_env(s_mask, V, eta_weight, dv_weight):
    binding_mgr = BindingManager(verbose=False)
    spec = Specification()
    spec.get_task_specification("Case2", s=s_mask, binding_manager=binding_mgr)

    ws = Workspace(size=(50, 40), target_mask=s_mask,
                   num_drones=3, num_gvs=4, num_humans=2, margin=4)
    binding_mgr.agents_by_type = ws.agents

    labeler = Labeler(spec)

    allocator = RLAllocator(spec, ws.agents, binding_mgr, labeler, ws,
                            value_bank=V, eta_weight=eta_weight, dv_weight=dv_weight)
    sim = Simulation(spec, ws, allocator, labeler)
    return spec, ws, labeler, binding_mgr, allocator, sim

def is_oversight_ap(ap: str) -> bool:
    # Adjust if your oversight set differs
    return ap.startswith("p_priority") or ap.startswith("p_triage") or ap.startswith("p_verify")

def oversight_delay_penalty(ap: str, t_assign: float, t_done: float, eta: float = 0.0) -> float:
    # Plug your real delay metric here if you already log it; default 0
    if eta <= 0.0:  # penalty disabled
        return 0.0
    # Example: flat penalty proportional to elapsed time unattended
    return eta * max(0.0, t_done - t_assign)

def parse_target_id(ap: str) -> int:
    try:
        # e.g., p_nav_7_1_1_0 -> 7 (third token)
        return int(ap.split("_")[2])
    except Exception:
        return -1


# -------------------- training runner --------------------

def run_train_test(
    episodes: int = 10,
    steps_per_ep: int = 2000,
    s_dim: int = 3,
    eta_weight: float = 1.0,     # weight on ETA in score
    dv_weight: float = 1.0,      # weight on ΔV in score
    oversight_eta: float = 0.0,  # >0 to activate oversight penalty in tau
    model_ckpt: str = "checkpoints/value_bank.pt",
    seed: int = 0,
):
    """
    Patched training loop:
      - Adds timed world events (FIRE/SURVIVOR/ATM) and verify-gate selection.
      - Pads workspace indices so tid lookups are always safe (no core edits).
      - Records TD transitions using GROUP DFAs (q_before/q_after).
    """

    os.makedirs(os.path.dirname(model_ckpt) or ".", exist_ok=True)
    rng = np.random.RandomState(seed)
    DT = 1.0  # env timestep

    # ---- helpers (script-side, no core changes) -------------------------------

    # Events like your old loop
    class EventScheduler:
        def __init__(self,
                     fire_times=(30.0, 45.0, 80.0, 110.0, 150.0, 250.0, 300.0),
                     surv_times=(25.0, 55.0, 70.0, 100.0, 156.0, 169.3, 264.5),
                     atm_times =(35.0, 65.0, 90.0, 120.0, 154.2, 185.3, 210.4, 290.0),
                     rng=None):
            self.fire_times = list(fire_times)
            self.surv_times = list(surv_times)
            self.atm_times  = list(atm_times)
            self.i_fire = self.i_surv = self.i_atm = 0
            self.rng = rng

        def _choose_tid_for_fire(self, ws):
            try:
                all_agents = ws.agents.get("drones", []) + ws.agents.get("gvs", [])
            except Exception:
                all_agents = []
            locs = getattr(ws, "target_locations", [])
            if not locs:
                return 0
            if all_agents:
                a = self.rng.choice(all_agents) if self.rng is not None else np.random.choice(all_agents)
                dists = [np.linalg.norm(np.array(loc) - np.array(a.pos[:2])) for loc in locs]
                k = max(1, len(dists) // 3)
                cand = np.argsort(dists)[:k]
                choices = [tid for tid in cand if ws.get_target_priority(tid) < 2] or list(range(len(locs)))
                return int(self.rng.choice(choices) if self.rng is not None else np.random.choice(choices))
            return int(self.rng.integers(len(locs)) if self.rng is not None else 0)

        def tick(self, t_now: float, ws, labeler):
            # FIRE → bump priority then emit firemsg
            if self.i_fire < len(self.fire_times) and t_now >= self.fire_times[self.i_fire]:
                tid = self._choose_tid_for_fire(ws)
                required = int((self.rng.choice([2, 2, 1]) if self.rng is not None else np.random.choice([2, 2, 1])))
                ws.set_target_priority(tid, required)
                labeler.advance({"p_firemsg_0_0_0_0"})
                self.i_fire += 1
            # SURVIVORMSG
            if self.i_surv < len(self.surv_times) and t_now >= self.surv_times[self.i_surv]:
                labeler.advance({"p_survivormsg_0_0_0_0"})
                self.i_surv += 1
            # ATMMSG
            if self.i_atm < len(self.atm_times) and t_now >= self.atm_times[self.i_atm]:
                labeler.advance({"p_atmmsg_0_0_0_0"})
                self.i_atm += 1

        def on_completed(self, ap: str, labeler):
            # When a verify finishes, choose found/notfound gate (80/20)
            if ap.startswith("p_verify"):
                try:
                    tid = ap.split("_")[2]
                except Exception:
                    return
                gate = (f"p_foundgate_{tid}"
                        if (self.rng.random() if self.rng is not None else np.random.random()) < 0.8
                        else f"p_notfoundgate_{tid}")
                labeler.chosen_gate_per_group[tid] = gate
                labeler.advance({gate})

    # Avoid IndexError when active_k < n_regions by padding compact lists
    def _pad_ws_indices(ws, n_regions: int):
        locs = list(getattr(ws, "target_locations", []))
        if len(locs) < n_regions:
            filler = locs[0] if locs else (0, 0)
            locs.extend([filler] * (n_regions - len(locs)))
            ws.target_locations = locs
        dop = dict(getattr(ws, "dropoff_locations", {}))
        if dop:
            example = next(iter(dop.values()))
            for k in range(n_regions):
                if k not in dop:
                    dop[k] = example
            ws.dropoff_locations = dop

    # Build a simple all-active mask (edit if you randomize)
    s_mask = [1] * 15

    # ---- RL components --------------------------------------------------------
    V = ValueBank(ValueNetConfig(s_dim=s_dim))                                  # :contentReference[oaicite:4]{index=4}
    replay = APReplay(capacity=100_000, seed=seed)                               # :contentReference[oaicite:5]{index=5}
    trainer = CriticTrainer(V, replay, TrainerCfg(batch_size=64, alpha=1e-3))    # :contentReference[oaicite:6]{index=6}

    # Dispatch cache: ap -> (group, q_before, s_vec, t_assign, agent_label)
    dispatch = {}

    # ---- Episodes -------------------------------------------------------------
    for ep in range(1, episodes + 1):
        # Fresh env per episode (reuse your helper)
        spec, ws, labeler, binding_mgr, allocator, sim = make_env(s_mask, V, eta_weight, dv_weight)  # :contentReference[oaicite:7]{index=7}
        _pad_ws_indices(ws, n_regions=len(s_mask))  # script-side safety shim
        dispatch.clear()

        # Timed events
        scheduler = EventScheduler(rng=rng)

        ep_completed = 0
        ep_assigned = 0

        for step in range(steps_per_ep):
            out = sim.step(DT, mode="sim", verbose=False)
            unlocked, assignments, completed = out["unlocked"], out["assignments"], out["completed"]

            # Inject world events
            t_now = float(getattr(sim, "time", step * DT))
            scheduler.tick(t_now, ws, labeler)

            # Cache dispatch for new assignments (GROUP state, not AP)
            for agent, ap in assignments.items():
                if ap not in dispatch:
                    group = binding_mgr.task_to_group.get(ap, None)  # group DFA key
                    q_before = labeler.states.get(group)
                    s_vec = build_s_vector(ws, ap)                    # :contentReference[oaicite:8]{index=8}
                    who = getattr(agent, "label", None) or getattr(agent, "name", None) or f"agent_{getattr(agent, 'id', id(agent))}"
                    dispatch[ap] = (group, q_before, s_vec, t_now, who)
                    ep_assigned += 1

            # On completion → choose gates, push TD samples, online update
            for ap in completed:
                scheduler.on_completed(ap, labeler)  # verify gate choice
                info = dispatch.pop(ap, None)
                if info is None:
                    continue
                group, q_before, s_vec, t_assign, who = info
                t_done = t_now
                tau = float(t_done - t_assign)
                if is_oversight_ap(ap):
                    tau += oversight_delay_penalty(ap, t_assign, t_done, eta=oversight_eta)
                q_after = labeler.states.get(group)

                # TD sample for critic
                replay.push(ap, q_before, q_after, s_vec, tau,
                            meta={"ep": ep, "step": step, "agent": who, "tid": parse_target_id(ap)})  # :contentReference[oaicite:9]{index=9}
                ep_completed += 1
                trainer.update_online(n_recent=1)  # immediate TD(0) step                      :contentReference[oaicite:10]{index=10}

            # Optional periodic minibatch updates
            if step % 50 == 0 and len(replay) >= 64:
                trainer.update()  # sampled batch TD(0)                                        :contentReference[oaicite:11]{index=11}

            if getattr(sim, "done", False):
                break
            
            # Check termination
            if labeler.all_completed() and ws.all_mobile_agents_at_base():
                print(f"[t={t_now:.2f}] Mission completed!")
                break

        # Save critic each episode
        V.save(model_ckpt)  # value bank checkpoint                                           :contentReference[oaicite:12]{index=12}
        print(f"EP {ep:03d}: assigned={ep_assigned} completed={ep_completed} saved={model_ckpt}")

    print("Training finished.")

    # ---- Quick eval pass (no learning) ----------------------------------------
    print("Testing learned allocator (no learning)...")
    spec, ws, labeler, binding_mgr, allocator, sim = make_env(s_mask, V, eta_weight, dv_weight)      # :contentReference[oaicite:13]{index=13}
    _pad_ws_indices(ws, n_regions=len(s_mask))
    for step in range(min(steps_per_ep, 2000)):
        out = sim.step(DT, mode="sim", verbose=False)
        if getattr(sim, "done", False):
            break
    print("Test run complete.")


# -------------------- CLI --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--s-dim", type=int, default=3)
    p.add_argument("--eta-weight", type=float, default=1.0)
    p.add_argument("--dv-weight", type=float, default=1.0)
    p.add_argument("--oversight-eta", type=float, default=0.0)
    p.add_argument("--ckpt", type=str, default="checkpoints/value_bank.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_train_test(
        episodes=args.episodes,
        steps_per_ep=args.steps,
        s_dim=args.s_dim,
        eta_weight=args.eta_weight,
        dv_weight=args.dv_weight,
        oversight_eta=args.oversight_eta,
        model_ckpt=args.ckpt,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
