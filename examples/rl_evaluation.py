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
                   labeler: Labeler,
                   scheduler,                          # EventScheduler
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

    # --- Lightweight utilization / score state (to make logger plots meaningful) ---
    SLIDING_WINDOW = 60.0
    SA_DECAY = 0.98
    human_SA = {h.label: 0.0 for h in ws.agents.get("humans", [])}

    # Seed busy/idle history for utilization; default utilization=0.0
    for h in ws.agents.get("humans", []):
        h.util_history = [(0.0, "idle")]
        h._last_state = "idle"
        h.utilization = 0.0

    frames_recorded = 0

    for step in range(steps):
        out = sim.step(dt, mode="sim", verbose=False)
        assignments = out.get("assignments", {})
        completed   = out.get("completed", [])

        # current sim time (sim may not track 'time', so compute from loop)
        t_now = float(getattr(sim, "time", step * dt))

        # ---- drive world events + choose verify gates so DFAs can accept ----
        if scheduler is not None:
            scheduler.tick(t_now, ws, labeler)
            for ap in completed:
                scheduler.on_completed(ap, labeler)

        # --- update human SA (decay + credit on completion of symbolic APs) ---
        for lbl in human_SA:
            human_SA[lbl] *= SA_DECAY
        for ap in completed:
            if ap.startswith(("p_verify", "p_priority", "p_triage", "p_atmconfirm")):
                # Credit to the human who appears to be handling this AP now
                for h in ws.agents.get("humans", []):
                    if assignments.get(h) == ap or getattr(h, "current_symbolic_task", None) == ap:
                        human_SA[h.label] += 1.0

        # --- maintain busy/idle history + sliding-window utilization ---
        for h in ws.agents.get("humans", []):
            is_busy = (assignments.get(h) is not None)
            new_state = "busy" if is_busy else "idle"
            if new_state != h._last_state:
                h.util_history.append((t_now, new_state))
                h._last_state = new_state

            # prune to window, keeping the last event before window start
            t0 = t_now - SLIDING_WINDOW
            ev = h.util_history
            older = [e for e in ev if e[0] < t0]
            newer = [e for e in ev if e[0] >= t0]
            h.util_history = ([max(older, key=lambda x: x[0])] if older else []) + newer

            # integrate busy time over [t0, t_now]
            busy_time = 0.0
            ev2 = h.util_history
            last_t, last_s = (ev2[0] if ev2 else (t_now, "idle"))
            last_t = max(last_t, t0)
            for (tt, ss) in ev2[1:] + [(t_now, h._last_state)]:
                seg_start = max(last_t, t0)
                seg_end = min(tt, t_now)
                if seg_end > seg_start and last_s == "busy":
                    busy_time += (seg_end - seg_start)
                last_t, last_s = tt, ss
            h.utilization = 100.0 * (busy_time / SLIDING_WINDOW if SLIDING_WINDOW > 0 else 0.0)

        # sample logger sparsely
        if logger is not None and (step % log_every) == 0:
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

        # Optional short debug trace (uncomment if needed)
        # print(f"[t={t_now:.1f}] unlocked={len(out.get('unlocked', []))} "
        #       f"assigned={[f'{a.label}->{ap}' for a, ap in assignments.items()]} "
        #       f"completed={len(completed)}")

        # termination conditions
        if getattr(sim, "done", False):
            break
        if labeler.all_completed() and ws.all_mobile_agents_at_base():
            print(f"[t={t_now:.2f}] Mission completed!")
            break

    return frames_recorded


def _pad_ws_indices(ws, n_regions: int):
    """Pad target and dropoff indices so (mask-id) == (index) is always safe."""
    # targets
    locs = list(getattr(ws, "target_locations", []))
    if len(locs) < n_regions:
        filler = locs[0] if locs else (0, 0)
        locs.extend([filler] * (n_regions - len(locs)))
        ws.target_locations = locs

    # dropoffs (dict keyed by tid)
    dop = dict(getattr(ws, "dropoff_locations", {}))
    if dop:
        example = next(iter(dop.values()))
        for k in range(n_regions):
            if k not in dop:
                dop[k] = example
        ws.dropoff_locations = dop


def make_env(s_mask, V: ValueBank, eta_weight: float = 1.0, dv_weight: float = 1.0):
    bm = BindingManager(verbose=False)
    spec = Specification()
    spec.get_task_specification("Case2", s=s_mask, binding_manager=bm)

    ws = Workspace(size=(50, 40), target_mask=s_mask,
                   num_drones=3, num_gvs=4, num_humans=2, margin=4)
    _pad_ws_indices(ws, n_regions=len(s_mask))
    bm.agents_by_type = ws.agents

    labeler = Labeler(spec)
    alloc = RLAllocator(spec, ws.agents, bm, labeler, ws,
                        value_bank=V, eta_weight=eta_weight, dv_weight=dv_weight)
    sim = Simulation(spec, ws, alloc, labeler)
    return spec, ws, labeler, bm, alloc, sim


# ---- Timed event scheduler (mirrors your dev loop) ---------------------------
class EventScheduler:
    def __init__(self,
                #  fire_times=(30.0, 45.0, 80.0, 110.0, 150.0, 250.0, 300.0),
                #  surv_times=(25.0, 55.0, 70.0, 100.0, 156.0, 169.3, 264.5),
                #  atm_times =(35.0, 65.0, 90.0, 120.0, 154.2, 185.3, 210.4, 290.0),
                 fire_times=(),
                 surv_times=(),
                 atm_times =(),
                 rng=None):
        self.fire_times = list(fire_times)
        self.surv_times = list(surv_times)
        self.atm_times  = list(atm_times)
        self.i_fire = self.i_surv = self.i_atm = 0
        self.rng = rng

    def _choose_tid_for_fire(self, ws):
        agents = ws.agents.get("drones", []) + ws.agents.get("gvs", [])
        locs = getattr(ws, "target_locations", [])
        if not locs:
            return 0
        if agents:
            a = self.rng.choice(agents) if self.rng is not None else np.random.choice(agents)
            dists = [np.linalg.norm(np.array(loc) - np.array(a.pos[:2])) for loc in locs]
            k = max(1, len(dists) // 3)
            cand = np.argsort(dists)[:k]
            choices = [tid for tid in cand if ws.get_target_priority(tid) < 2] or list(range(len(locs)))
            return int(self.rng.choice(choices) if self.rng is not None else np.random.choice(choices))
        return 0

    def tick(self, t_now: float, ws, labeler):
        if self.i_fire < len(self.fire_times) and t_now >= self.fire_times[self.i_fire]:
            tid = self._choose_tid_for_fire(ws)
            required = int((self.rng.choice([2,2,1]) if self.rng is not None else np.random.choice([2,2,1])))
            ws.set_target_priority(tid, required)
            labeler.advance({"p_firemsg_0_0_0_0"})
            self.i_fire += 1
        if self.i_surv < len(self.surv_times) and t_now >= self.surv_times[self.i_surv]:
            labeler.advance({"p_survivormsg_0_0_0_0"})
            self.i_surv += 1
        if self.i_atm < len(self.atm_times) and t_now >= self.atm_times[self.i_atm]:
            labeler.advance({"p_atmmsg_0_0_0_0"})
            self.i_atm += 1

    def on_completed(self, ap: str, labeler):
        # choose verify gate (80/20) so the group DFA can accept
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

    # 2) Randomize active regions once (keep as requested)
    rng = np.random.RandomState(args.seed)
    s_mask = [0] * args.n_regions
    for i in rng.choice(args.n_regions, size=min(args.active_k, args.n_regions), replace=False):
        s_mask[i] = 1

    # 3) Build env
    spec, ws, labeler, bm, alloc, sim = make_env(s_mask, V, args.eta_weight, args.dv_weight)

    # 4) Logger (optional)
    logger = GuiEpisodeLogger(Path(args.logdir), ws.get_all_agents())

    # 4.5) Event scheduler (mirrors FIRE/SURV/ATM logic)
    scheduler = EventScheduler(rng=np.random.RandomState(args.seed))

    # 5) Run episode with gradients disabled (critical for speed)
    if torch is not None:
        with torch.no_grad():
            frames = record_episode(sim, ws, labeler, scheduler,
                                    steps=args.steps, dt=args.dt,
                                    logger=logger,
                                    log_every=args.log_every,
                                    frame_skip=args.frame_skip)
    else:
        frames = record_episode(sim, ws, labeler, scheduler,
                                steps=args.steps, dt=args.dt,
                                logger=logger,
                                log_every=args.log_every,
                                frame_skip=args.frame_skip)

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
