from __future__ import annotations
import argparse, sys, os
from pathlib import Path

# --- env vars BEFORE imports ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Project root
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

# Optional: cap torch threads & disable grad for inference
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    torch = None


def record_episode(sim: Simulation,
                   ws: Workspace,
                   steps: int,
                   dt: float,
                   logger: GuiEpisodeLogger | None = None,
                   log_every: int = 1,
                   frame_skip: int = 1) -> int:
    """
    Run one episode and (optionally) log + record trajectories.
    - logger sampled every `log_every` steps
    - trajectories recorded every `frame_skip` steps (for playback GIF)
    Returns the number of recorded frames.
    """
    # Prepare playback arrays
    for a in ws.get_all_agents():
        a.traj = []
        a.progress_traj = []

    frames_recorded = 0

    for step in range(steps):
        out = sim.step(dt, mode="sim", verbose=False)
        assignments = out.get("assignments", {})
        completed   = out.get("completed", [])

        # sample logger sparsely
        if logger is not None and (step % log_every) == 0:
            t_now = getattr(sim, "time", step * dt)
            logger.note_step(t_now, assignments, completed, ws.agents.get("humans", []))

        # record positions/progress sparsely for GIF playback
        if (step % frame_skip) == 0:
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


def _pad_ws_targets(ws, n_regions: int):
    locs = list(getattr(ws, "target_locations", []))
    if len(locs) < n_regions:
        filler = locs[0] if locs else (0, 0)
        locs.extend([filler] * (n_regions - len(locs)))
        ws.target_locations = locs


def make_env(s_mask, V: ValueBank, eta_weight: float = 1.0, dv_weight: float = 1.0):
    bm = BindingManager(verbose=False)
    spec = Specification()
    spec.get_task_specification("Case2", s=s_mask, binding_manager=bm)

    ws = Workspace(size=(50, 40), target_mask=s_mask,
                   num_drones=3, num_gvs=4, num_humans=2, margin=4)
    _pad_ws_targets(ws, n_regions=len(s_mask))
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
    ap.add_argument("--dt", type=float, default=1.0, help="simulation timestep")
    ap.add_argument("--s-dim", type=int, default=3)
    ap.add_argument("--eta-weight", type=float, default=1.0)
    ap.add_argument("--dv-weight", type=float, default=1.0)
    ap.add_argument("--n-regions", type=int, default=15)
    ap.add_argument("--active-k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--gif", type=str, default="run.gif")
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--logdir", type=str, default="runs")
    ap.add_argument("--log-every", type=int, default=1, help="log GUI metrics every K ticks")
    ap.add_argument("--frame-skip", type=int, default=4, help="record 1 playback frame every K ticks")
    args = ap.parse_args()

    # 1) Load trained values (in eval mode; grad off)
    V = ValueBank(ValueNetConfig(s_dim=args.s_dim))
    V.load(args.ckpt)
    if hasattr(V, "eval"):
        V.eval()

    # 2) Randomize active regions once
    rng = np.random.RandomState(args.seed)
    s_mask = [0] * args.n_regions
    for i in rng.choice(args.n_regions, size=min(args.active_k, args.n_regions), replace=False):
        s_mask[i] = 1

    # 3) Build env
    spec, ws, labeler, bm, alloc, sim = make_env(s_mask, V, args.eta_weight, args.dv_weight)

    # 4) Logger (optional)
    logger = GuiEpisodeLogger(Path(args.logdir), ws.get_all_agents())

    # 5) Run episode with gradients disabled (critical for speed)
    if torch is not None:
        with torch.no_grad():
            frames = record_episode(sim, ws, steps=args.steps, dt=args.dt,
                                    logger=logger, log_every=args.log_every, frame_skip=args.frame_skip)
    else:
        frames = record_episode(sim, ws, steps=args.steps, dt=args.dt,
                                logger=logger, log_every=args.log_every, frame_skip=args.frame_skip)

    # 6) Optional GIF (playback). Each frame represents (dt * frame_skip) seconds.
    if not args.no_gif:
        interval_ms = 100
        animate_workspace(ws, steps=frames, interval=interval_ms,
                          sim=None, dt=args.dt, record=True, output_path=args.gif)
        print(f"Saved GIF: {args.gif}  (frames={frames}, frame_skip={args.frame_skip}, interval={interval_ms}ms)")

    # 7) Save episode metrics/summary
    mct = getattr(sim, "time", frames * args.dt * args.frame_skip)
    csv_path, png_path = logger.save_all(mct=mct, show=False)
    print(f"Saved metrics CSV: {csv_path}")
    print(f"Saved episode summary PNG: {png_path}")


if __name__ == "__main__":
    main()
