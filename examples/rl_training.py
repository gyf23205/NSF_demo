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
    os.makedirs(os.path.dirname(model_ckpt) or ".", exist_ok=True)
    rng = np.random.RandomState(seed)

    DT = 1.0  # env timestep

    # --- 1) Build spec, binding, workspace, labeler (match repo pattern) ---
    # (a) binding manager
    binding_mgr = BindingManager(verbose=False)
    # (b) pick a region mask (same style as rl_training.py)
    s_mask = [1] * 15
    # (c) spec with DAG+automata populated
    spec = Specification()
    spec.get_task_specification("Case2", s=s_mask, binding_manager=binding_mgr)
    # (d) workspace + agents
    ws = Workspace(size=(50, 40), target_mask=s_mask, num_drones=3, num_gvs=4, num_humans=2, margin=4)
    binding_mgr.agents_by_type = ws.agents
    # (e) labeler after spec has dag/automata
    labeler = Labeler(spec)

    # --- 2) Value function + replay + trainer ---
    V = ValueBank(ValueNetConfig(s_dim=s_dim))
    replay = APReplay(capacity=100_000, seed=seed)
    trainer = CriticTrainer(V, replay, TrainerCfg(batch_size=64, alpha=1e-3, updates_per_call=1))

    # --- 3) Allocator + Simulation ---
    policy = MaskedMatchPolicy()            # greedy masked matcher (no NN yet)
    # RLAllocator reuses your RandomAllocator helpers & binding logic internally
    allocator = RLAllocator(spec, ws.agents, labeler.binding_manager, labeler, ws,
                            value_bank=V, eta_weight=eta_weight, dv_weight=dv_weight)
    sim = Simulation(spec, ws, allocator, labeler)

    # --- 4) Dispatch cache for TD targets ---
    # ap -> (q_before, s_vec, t_assign, agent_label)
    dispatch = {}

    # --- 5) Training loop ---
    for ep in range(1, episodes + 1):
        # Recreate fresh env each episode (since Simulation.reset() isn’t available)
        spec, ws, labeler, binding_mgr, allocator, sim = make_env(s_mask, V, eta_weight, dv_weight)
        dispatch.clear()

        ep_return = 0.0
        ep_completed = 0
        ep_assigned = 0

        for step in range(steps_per_ep):
            out = sim.step(DT)
            unlocked, assignments, completed = out["unlocked"], out["assignments"], out["completed"]

            # Cache dispatch for newly assigned APs
            for agent, ap in assignments.items():
                if ap not in dispatch:
                    q_before = labeler.states.get(ap)
                    s_vec = build_s_vector(ws, ap)
                    t_assign = getattr(sim, "time", step)
                    who = getattr(agent, "label", None) or getattr(agent, "name", None) or f"agent_{getattr(agent, 'id', id(agent))}"
                    dispatch[ap] = (q_before, s_vec, t_assign, who)
                    ep_assigned += 1

            # On completion → push sample + online TD
            for ap in completed:
                info = dispatch.pop(ap, None)
                if info is None:
                    continue
                q_before, s_vec, t_assign, who = info
                t_done = getattr(sim, "time", step)
                tau = float(t_done - t_assign)
                if is_oversight_ap(ap):
                    tau += oversight_delay_penalty(ap, t_assign, t_done, eta=oversight_eta)
                q_after = labeler.states.get(ap)

                replay.push(ap, q_before, q_after, s_vec, tau,
                            meta={"ep": ep, "step": step, "agent": who, "tid": parse_target_id(ap)})
                ep_completed += 1
                trainer.update_online(n_recent=1)

            if step % 50 == 0 and len(replay) >= 64:
                stats = trainer.update()
                print(f"[ep {ep:03d} step {step:04d}] buf={len(replay)} td2_ma={stats['td2_ma']:.4f}")

            if getattr(sim, "done", False):
                break

            ep_return -= float(getattr(sim, "dt", 1.0))

        V.save(model_ckpt)
        print(f"EP {ep:03d}: assigned={ep_assigned} completed={ep_completed} saved={model_ckpt}")

    print("Training finished.")

    # --- 6) Quick test run with learned V (no updates) ---
    print("Testing learned allocator (no learning)...")
    spec, ws, labeler, binding_mgr, allocator, sim = make_env(s_mask, V, eta_weight, dv_weight)
    dispatch.clear()
    for step in range(min(steps_per_ep, 2000)):
        out = sim.step(DT)
        unlocked, assignments, completed = out["unlocked"], out["assignments"], out["completed"]
        if getattr(sim, "done", False):
            break
    print("Test run complete.")


# -------------------- CLI --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
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
