from __future__ import annotations
import argparse, sys, os
from pathlib import Path

# --- env vars BEFORE any scientific imports (fix duplicate OpenMP on Windows) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Make project root importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from ltl_core.specification import Specification
from ltl_core.labeler import Labeler
from ltl_core.workspace import Workspace
from ltl_core.simulation import Simulation
from ltl_core.binding_manager import BindingManager
from ltl_core.value_fn import ValueNetConfig, ValueBank
from ltl_core.visualization import animate_workspace, GuiEpisodeLogger
from rl.allocator_rl import RLAllocator

DT = 1.0  # simulation timestep (seconds)

def record_episode(sim: Simulation, ws: Workspace, steps: int, dt: float, logger: GuiEpisodeLogger | None = None) -> int:
    """
    Run one episode (no visualization), record per-agent trajectories for playback,
    and optionally log metrics to GuiEpisodeLogger. Returns the number of frames recorded.
    """
    # Prepare playback arrays on each agent
    for a in ws.get_all_agents():
        a.traj = []
        a.progress_traj = []

    frames_recorded = 0
    for step in range(steps):
        out = sim.step(dt)  # your Simulation.step returns a dict
        unlocked = out.get("unlocked")
        assignments = out.get("assignments", {})
        completed = out.get("completed", [])

        # Log GUI metrics (optional)
        if logger is not None:
            t_now = getattr(sim, "time", step * dt)
            logger.note_step(t_now, assignments, completed, ws.agents.get("humans", []))

        # Record positions & progress for playback
        for a in ws.get_all_agents():
            r, c = a.pos[:2]
            a.traj.append((r, c))
            if hasattr(a, "current_symbolic_task") and a.current_symbolic_task:
                p_val = float(a.get_progress(a.current_symbolic_task))
            else:
                p_val = 0.0
            a.progress_traj.append(p_val)

        frames_recorded += 1
        if getattr(sim, "done", False):
            break

    return frames_recorded

def make_env(s_mask, V: ValueBank, eta_weight: float = 1.0, dv_weight: float = 1.0):
    """
    Build a fresh environment bound to the provided ValueBank.
    """
    bm = BindingManager(verbose=False)
    spec = Specification()
    spec.get_task_specification("Case2", s=s_mask, binding_manager=bm)

    ws = Workspace(size=(50, 40), target_mask=s_mask,
                   num_drones=3, num_gvs=4, num_humans=2, margin=4)
    bm.agents_by_type = ws.agents

    labeler = Labeler(spec)
    alloc = RLAllocator(spec, ws.agents, bm, labeler, ws,
                        value_bank=V, eta_weight=eta_weight, dv_weight=dv_weight)
    sim = Simulation(spec, ws, alloc, labeler)
    return spec, ws, labeler, bm, alloc, sim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/value_bank.pt")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--s-dim", type=int, default=3)
    ap.add_argument("--eta-weight", type=float, default=1.0)
    ap.add_argument("--dv-weight", type=float, default=1.0)
    ap.add_argument("--n-regions", type=int, default=15)
    ap.add_argument("--active-k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gif", type=str, default="run.gif")
    ap.add_argument("--logdir", type=str, default="runs")
    args = ap.parse_args()

    # 1) Load trained values
    V = ValueBank(ValueNetConfig(s_dim=args.s_dim))
    V.load(args.ckpt)

    # 2) Pick a randomized scenario (which regions are active)
    rng = np.random.RandomState(args.seed)
    s_mask = [0] * args.n_regions
    for i in rng.choice(args.n_regions, size=min(args.active_k, args.n_regions), replace=False):
        s_mask[i] = 1

    # 3) Build env
    spec, ws, labeler, bm, alloc, sim = make_env(s_mask, V, args.eta_weight, args.dv_weight)

    # 4) Set up logger
    logger = GuiEpisodeLogger(Path(args.logdir), ws.get_all_agents())

    # 5) RUN EPISODE (no viz), **record** trajectories + log metrics
    frames = record_episode(sim, ws, steps=args.steps, dt=DT, logger=logger)

    # 6) RENDER GIF IN PLAYBACK MODE (sim=None so it won't live-step)
    animate_workspace(ws, steps=frames, interval=100,
                      sim=None, dt=DT, record=True, output_path=args.gif)

    # 7) Save episode-level metrics & dashboard
    mct = getattr(sim, "time", frames * DT)
    csv_path, png_path = logger.save_all(mct=mct, show=False)

    print(f"Saved GIF: {args.gif}")
    print(f"Saved metrics CSV: {csv_path}")
    print(f"Saved episode summary PNG: {png_path}")

if __name__ == "__main__":
    main()
