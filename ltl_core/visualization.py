import re
import pygame
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
from ltl_core import env_ltl as env
from pathlib import Path
from ltl_core.specification import get_ap_prefix
import csv
import matplotlib.gridspec as gridspec


# Colors for timeline by AP prefix (extend as needed)
AP_COLORS = {
    "idle":       "#bdbdbd",
    "p_nav":      "#1f77b4",
    "p_scan":     "#2ca02c",
    "p_verify":   "#9467bd",
    "p_pickup":   "#ff7f0e",
    "p_dropoff":  "#d62728",
    "p_priority": "#8c564b",
    "p_triage":   "#e377c2",
    "p_atmconfirm":"#17becf",
    "env":        "#7f7f7f",
}

AP_ABBR = {
    "idle": "IDLE", "p_nav": "NAV", "p_scan": "SCAN", "p_verify": "VER",
    "p_pickup": "PICK", "p_dropoff": "DROP", "p_priority": "PRIO",
    "p_triage": "TRI", "p_atmconfirm": "ATM", "env": "ENV",
}


def draw_workspace(screen, ws, font=None, screensize=(1200, 900), cell_size=30):
    """
    Draws workspace elements (base, hospital, targets, agents, halos, labels) on the Pygame screen.
    """
    if font is None:
        font = pygame.font.SysFont('Arial', 20)
    screen.fill((255, 255, 255))  # Clear background

    # 1. Draw base area
    for (x, y) in ws.base_area:
        px, py = ws.meter_to_pixel((x, y + 1.0))
        rect = pygame.Rect(px, py, cell_size, cell_size)
        pygame.draw.rect(screen, (173, 216, 230), rect)  # light blue

    # 2. Draw hospital area
    for (x, y) in ws.hospital_area:
        px, py = ws.meter_to_pixel((x, y + 1.0))
        rect = pygame.Rect(px, py, cell_size, cell_size)
        pygame.draw.rect(screen, (255, 182, 193), rect)  # misty rose

    # 3. Draw targets
    for i, (x, y) in enumerate(ws.target_locations):
        px, py = ws.meter_to_pixel((x + 0.5, y + 0.5))
        pygame.draw.circle(screen, (255, 0, 0), (px, py), 6)
        label = font.render(f"T{i}", True, (0, 0, 0))
        screen.blit(label, (px - 8, py - 20))

    # 4. Draw agents and progress
    for agent in ws.get_all_agents():
        role_color = {
            "drones": (0, 0, 255),
            "gvs": (0, 128, 0),
            "humans": (255, 165, 0)
        }
        color = role_color.get(agent.role, (100, 100, 100))
        grid_x, grid_y = agent.pos[:2]
        px, py = ws.meter_to_pixel((grid_x + 0.5, grid_y + 0.5))
        pygame.draw.circle(screen, color, (px, py), 10)

        # Progress halo
        if hasattr(agent, "current_symbolic_task") and agent.current_symbolic_task:
            p_val = agent.get_progress(agent.current_symbolic_task)
            if p_val > 0:
                pygame.draw.circle(screen, (50, 205, 50), (px, py), int(14 + 10 * p_val), width=2)
                progress_text = font.render(f"{p_val:.1f}", True, (0, 100, 0))
                screen.blit(progress_text, (px - 10, py - 25))

        # Label
        if hasattr(agent, "label"):
            label = font.render(agent.label, True, (0, 0, 0))
            screen.blit(label, (px - 12, py + 12))


def _english_label(node: str) -> str:
    if node == 'p_0':     return 'SAR Mission'
    if node == 'p_101':   return 'Search and Rescue'
    if node == 'p_102':   return 'Supervision'
    parts = node.split('_')

    # Special case: p_101_aux_{region}
    if len(parts) >= 4 and parts[0] == 'p' and parts[1] == '101' and parts[2] == 'aux':
        return f"Region {parts[3]}"

    kind = parts[1]
    region = parts[2] if len(parts) > 2 else ''
    m = {
      'search':       'Search',
      'rescue':       'Rescue',
      'oversight':    'Oversight',
      'navscan':      'NavScan',
      'nav':          'Navigate',
      'scan':         'Scan',
      'verify':       'Verify',
      'foundgate':    'Found',
      'notfoundgate': 'Not Found',
      'skip':         'Skip',
      'pickup':       'Pick Up',
      'dropoff':      'Drop Off',
      'fire':         'Fire',
      'survivor':     'Survivor',
      'atm':          'Air Traffic',
      'firemsg':      'Fire Msg',
      'priority':     'Set Priority',
      'survivormsg':  'Survivor Msg',
      'triage':       'Triage',
      'atmmsg':       'ATM Msg',
      'atmconfirm':   'ATM Confirm',
    }
    label = m.get(kind, node)
    return f"{label} {region}".strip()


def draw_composite_hierarchy(spec, figsize=(10, 6)):
    """
    - Lock Level-3 (generalized functions) on one horizontal rank
      and lay them left→right in the order [search, rescue, skip, …]
      with *uniform* spacing.
    - Level-4 nodes keep Graphviz's vertical placement but take the
      x-position of their Level-3 parent so they stay centred.
    - Everything else (styling, centring of Level-1 / Level-2) unchanged.
    """
    # 1) Composite-only sub-graph
    G = spec.dag
    H = G.subgraph(G.graph['composite_names']).copy()

    # 2) Build an AGraph; force Level-3 into one rank
    A = to_agraph(H)
    lvl3_nodes = list(spec.hierarchy[2].keys())        # keys of Level-3 dict
    A.add_subgraph(lvl3_nodes, name="rank3", rank="same")
    A.layout(prog="dot")

    # 3) read back Graphviz positions
    pos = {
        n: tuple(map(float, A.get_node(n).attr['pos'].split(',')))
        for n in H.nodes()
    }

    # 4) --- Re-order Level-3 horizontally with *uniform* spacing -------------
    present_lvl3 = [n for n in lvl3_nodes if n in pos]
    if present_lvl3:
        x_min = min(pos[n][0] for n in present_lvl3)
        x_max = max(pos[n][0] for n in present_lvl3)
        step  = 0 if len(present_lvl3) == 1 else (x_max - x_min) / (len(present_lvl3) - 1)
        for k, node in enumerate(present_lvl3):
            _, y = pos[node]          # keep original y (same rank)
            pos[node] = (x_min + k * step, y)
    # ------------------------------------------------------------------------

    # 4.5) --- Order aux nodes by numeric suffix (e.g., aux_0 before aux_1) -----
    aux_nodes = sorted(
        [n for n in pos if "aux" in n],
        key=lambda x: int(re.search(r'aux_(\d+)', x).group(1)) if re.search(r'aux_(\d+)', x) else 0
    )

    if aux_nodes:
        x_aux_min = min(pos[n][0] for n in aux_nodes)
        x_aux_max = max(pos[n][0] for n in aux_nodes)
        step_aux = 0 if len(aux_nodes) == 1 else (x_aux_max - x_aux_min) / (len(aux_nodes) - 1)

    for k, node in enumerate(aux_nodes):
        _, y = pos[node]
        pos[node] = (x_aux_min + k * step_aux, y)
    # ----------------------------------------------------------------------------

    # 5) --- Improved Level-4 layout with vertical chains and x-distribution -----
    lvl4_dict = spec.hierarchy[3]
    lvl3_nodes = spec.hierarchy[2].keys()

    for parent in lvl3_nodes:
        if parent not in pos:
            continue

        # Get Level-4 children of this parent
        children = [c for c in G.successors(parent) if c in lvl4_dict and c in pos]
        if not children:
            continue

        # Build undirected graph among Level-4 children
        subG = nx.Graph()
        subG.add_nodes_from(children)
        for c in children:
            for succ in G.successors(c):
                if succ in children:
                    subG.add_edge(c, succ)

        components = list(nx.connected_components(subG))

        if len(components) == 1:
            # All children are connected: vertical stack at parent's x
            x = pos[parent][0]
            sorted_chain = sorted(components[0], key=lambda n: pos[n][1])
            y_start = min(pos[n][1] for n in sorted_chain)

            for i, node in enumerate(sorted_chain):
                pos[node] = (x, y_start + i * 40)
        else:
            # Multiple disconnected chains: spread horizontally around parent
            total_width = 80 * max(len(components) - 1, 1)
            x_center = pos[parent][0]
            x_start = x_center - total_width / 2

            for i, comp in enumerate(components):
                chain = list(comp)
                chain.sort(key=lambda n: pos[n][1])
                x = x_start + i * 80
                y_top = min(pos[n][1] for n in chain)

                for j, node in enumerate(chain):
                    pos[node] = (x, y_top + j * 40)
    # ------------------------------------------------------------------------

    # --- Fix: Symmetric placement of Level-2 under SAR Mission ----------------
    root = next(iter(spec.hierarchy[0].keys()))  # e.g., "p_0"

    lvl2_nodes = [
        n for n in spec.hierarchy[1].keys()
        if n in pos and "aux" not in n
    ]
    if len(lvl2_nodes) == 2 and root in pos:
        x_root = pos[root][0]
        y_vals = [pos[n][1] for n in lvl2_nodes]
        y_avg = sum(y_vals) / len(y_vals)
        offset = 120  # horizontal distance from center

        # Sort to preserve original left/right visual order
        sorted_lvl2 = sorted(lvl2_nodes, key=lambda n: pos[n][0])

        # Assign positions symmetrically around SAR Mission
        pos[sorted_lvl2[0]] = (x_root - offset, y_avg)
        pos[sorted_lvl2[1]] = (x_root + offset, y_avg)
    # --------------------------------------------------------------------------

    # 6) Draw -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    nx.draw_networkx_edges(H, pos, ax=ax, edge_color="gray", arrowsize=16)

    for node, (x, y) in pos.items():
        is_aux = "aux" in node.lower()
        facecolor = "lightgreen" if is_aux else "skyblue"

        ax.text(
            x, y, _english_label(node),
            ha="center", va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc=facecolor, ec="black", lw=1.0)
        )

    level_names = [
        "Functional Purpose",
        "Abstract Function",
        "Generalized Function",
        "Physical Function"
    ]
    xmin = min(x for x, _ in pos.values()) - 50
    for lvl, label in enumerate(level_names):
        ys = [pos[n][1] for n in spec.hierarchy[lvl].keys() if n in pos]
        if ys:
            ax.text(
                xmin, sum(ys) / len(ys), label,
                ha="right", va="center", fontsize=12, fontweight="bold"
            )

    plt.tight_layout()
    plt.show()


def draw_atomic_pairwise(G: nx.DiGraph, figsize=(8,6)):
    """
    Draw only the atomic‐AP subgraph (leaf nodes) with pairwise edges.
    """
    atomic = G.graph['atomic_names']
    H = G.subgraph(atomic).copy()
    pos = graphviz_layout(H, prog='dot')

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    nx.draw_networkx_edges(H, pos, ax=ax, arrowsize=12)
    for n,(x,y) in pos.items():
        ax.text(
            x, y, _english_label(n),
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', ec='black')
        )

    plt.tight_layout()
    plt.show()

    # ==== Multi-episode aggregation & box plots ===================================
def find_unified_csvs(log_dir, pattern="hat_episode_*_unified.csv", limit=None):
    """
    Return a list of unified CSV paths in log_dir (newest first).
    """
    from pathlib import Path
    paths = sorted(Path(log_dir).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if limit is not None:
        paths = paths[:int(limit)]
    return [str(p) for p in paths]


def _infer_humans_from_steps(df_steps):
    """Infer human labels from columns like H0_SA/H0_util."""
    labels = set()
    for c in df_steps.columns:
        if c.endswith("_SA"):
            labels.add(c[:-3])
        if c.endswith("_util"):
            labels.add(c[:-5])
    return sorted(labels)


def compute_episode_stats_from_csv(csv_path):
    """
    Returns a DataFrame with one row per human for this episode:
    cols: ['episode_id','csv_path','human','sa_mean','util_mean','idle_ratio_pct','overload_ratio_pct',
           'switches','distance']
    - sa_mean: mean of H*_SA over step table (as-is scale from CSV)
    - util_mean: mean of H*_util over step table (kept in % if CSV is 0–100)
    - idle/overload ratios: taken from agent table (converted to %)
    - switches, distance: taken from agent table if present (NaN if missing)
    """
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(csv_path)
    steps  = df[df["table"] == "step"].copy()
    agents = df[df["table"] == "agent"].copy()

    humans = _infer_humans_from_steps(steps) if not steps.empty else sorted(agents["agent"].dropna().unique())
    rows = []

    for h in humans:
        sa_mean = None
        util_mean = None
        if not steps.empty:
            sa_col   = f"{h}_SA"
            util_col = f"{h}_util"
            if sa_col in steps.columns:
                sa_mean = float(steps[sa_col].dropna().mean())
            if util_col in steps.columns:
                util_mean = float(steps[util_col].dropna().mean())

        # agent-level aggregates
        idle_pct = None
        ovl_pct  = None
        switches = None
        dist     = None
        if not agents.empty:
            arow = agents[agents["agent"] == h]
            if not arow.empty:
                if "idle_ratio" in arow.columns and pd.notna(arow["idle_ratio"].iloc[0]):
                    idle_pct = float(arow["idle_ratio"].iloc[0]) * 100.0
                if "overload_ratio" in arow.columns and pd.notna(arow["overload_ratio"].iloc[0]):
                    ovl_pct = float(arow["overload_ratio"].iloc[0]) * 100.0
                if "switches" in arow.columns and pd.notna(arow["switches"].iloc[0]):
                    switches = float(arow["switches"].iloc[0])
                if "distance" in arow.columns and pd.notna(arow["distance"].iloc[0]):
                    dist = float(arow["distance"].iloc[0])

        rows.append({
            "episode_id": Path(csv_path).stem,
            "csv_path": str(csv_path),
            "human": h,
            "sa_mean": sa_mean,
            "util_mean": util_mean,
            "idle_ratio_pct": idle_pct,
            "overload_ratio_pct": ovl_pct,
            "switches": switches,
            "distance": dist,
        })

    return pd.DataFrame(rows)


def build_multi_episode_table(csv_paths):
    """
    Concatenate per-episode stats for all given CSVs.
    """
    import pandas as pd
    frames = []
    for p in csv_paths:
        try:
            frames.append(compute_episode_stats_from_csv(p))
        except Exception as e:
            print(f"[viz] Skipped {p}: {e}")
    if not frames:
        return pd.DataFrame(columns=["episode_id","csv_path","human","sa_mean","util_mean",
                                     "idle_ratio_pct","overload_ratio_pct","switches","distance"])
    return pd.concat(frames, ignore_index=True)


def _box(ax, data_by_label, title, ylabel):
    """
    Helper to draw a single boxplot for dict {label: list_of_values}.
    """
    import matplotlib.pyplot as plt
    labels = list(data_by_label.keys())
    series = [data_by_label[k] for k in labels]
    bp = ax.boxplot(series, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)


def plot_boxplots_across_episodes(csv_paths=None, log_dir=None, limit=None, show=True, save_dir=None):
    """
    Box plots across episodes for:
      - Average SA score (per human)
      - Average utilization (per human)
      - Idle ratio (per human, %)
      - Overload ratio (per human, %)

    You can pass either csv_paths OR a log_dir (it will scan *_unified.csv).
    If save_dir is provided, figures are saved as PNGs there.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if csv_paths is None:
        if log_dir is None:
            print("[viz] Provide csv_paths or log_dir.")
            return
        csv_paths = find_unified_csvs(log_dir, limit=limit)

    stats = build_multi_episode_table(csv_paths)
    if stats.empty:
        print("[viz] No episode stats available.")
        return

    # Group by human
    humans = sorted(stats["human"].dropna().unique())

    def collect(metric):
        d = {}
        for h in humans:
            vals = stats.loc[stats["human"] == h, metric].dropna().tolist()
            if vals:
                d[h] = vals
        return d

    plots = [
        ("sa_mean", "Average Situation Awareness by Human", "SA (avg)"),
        ("util_mean", "Average Utilization by Human", "Utilization [%]"),
        ("idle_ratio_pct", "Idle Ratio by Human", "Idle ratio [%]"),
        ("overload_ratio_pct", "Overload Ratio by Human", "Overload ratio [%]"),
    ]

    figs = []
    for metric, title, ylabel in plots:
        data = collect(metric)
        if not data:
            print(f"[viz] Skipping {metric}: no data.")
            continue
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        _box(ax, data, title + f"  (N={len(csv_paths)} episodes)", ylabel)
        figs.append((metric, fig))

    # Optional extras you might find interesting: switches & distance
    # Uncomment if you want these as well.
    #
    # for metric, title, ylabel in [
    #     ("switches", "Task Switches by Human", "Switch count"),
    #     ("distance", "Distance Traveled by Human", "Distance [grid units]"),
    # ]:
    #     data = collect(metric)
    #     if data:
    #         fig, ax = plt.subplots(figsize=(8.4, 4.8))
    #         _box(ax, data, title + f"  (N={len(csv_paths)} episodes)", ylabel)
    #         figs.append((metric, fig))

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for metric, fig in figs:
            fig.savefig(os.path.join(save_dir, f"box_{metric}.png"), dpi=150, bbox_inches="tight")

    if show:
        import matplotlib.pyplot as plt
        plt.show()


def plot_ap_latency_boxplots(csv_paths=None, log_dir=None, limit=None, show=True, save_dir=None):
    """
    Optional: Box plots for AP latencies pooled across episodes:
      - assign→complete
      - unlock→assign
    Useful to understand scheduling/coordination delays.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if csv_paths is None:
        if log_dir is None:
            print("[viz] Provide csv_paths or log_dir.")
            return
        csv_paths = find_unified_csvs(log_dir, limit=limit)

    vals_assign_complete = []
    vals_unlock_assign = []

    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            ap = df[df["table"] == "ap_event"].copy()
            if not ap.empty:
                if "lat_assign_to_complete" in ap.columns:
                    vals_assign_complete += ap["lat_assign_to_complete"].dropna().tolist()
                if "lat_unlock_to_assign" in ap.columns:
                    vals_unlock_assign += ap["lat_unlock_to_assign"].dropna().tolist()
        except Exception as e:
            print(f"[viz] Skipped {p}: {e}")

    data = {
        "Assign→Complete [s]": vals_assign_complete,
        "Unlock→Assign [s]": vals_unlock_assign,
    }
    data = {k: v for k, v in data.items() if v}

    if not data:
        print("[viz] No latency data found.")
        return

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    _box(ax, data, f"AP Latencies (pooled)  (N={len(csv_paths)} episodes)", "Seconds")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "box_ap_latencies.png"), dpi=150, bbox_inches="tight")

    if show:
        import matplotlib.pyplot as plt
        plt.show()
# ==============================================================================


# ---- Metrics dashboard helpers ------------------------------------------------
def find_latest_metrics_csv(log_dir, pattern="hat_episode_*_unified.csv"):
    """
    Return the newest unified-metrics CSV path in log_dir, or None if not found.
    """
    from pathlib import Path
    paths = sorted(Path(log_dir).glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(paths[0]) if paths else None


def plot_episode_metrics(csv_path=None, log_dir=None, human_labels=None, show=True):
    """
    Create three figures from the unified metrics CSV:
      1) Time vs H*_SA
      2) Time vs H*_util   (0–100%)
      3) Bar chart: Idle vs Overload ratios per human (episode aggregates)

    Notes:
    - If csv_path is None, will search log_dir for the newest *_unified.csv
    - 'show' controls plt.show(); leave False if the caller will show later.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    # Resolve CSV path
    if csv_path is None:
        if log_dir is None:
            print("[viz] No csv_path or log_dir provided.")
            return
        csv_path = find_latest_metrics_csv(log_dir)
    if not csv_path:
        print("[viz] No metrics CSV found.")
        return

    df = pd.read_csv(csv_path)
    if "table" not in df.columns:
        print(f"[viz] CSV missing 'table' column: {csv_path}")
        return

    steps  = df[df["table"] == "step"].copy()
    agents = df[df["table"] == "agent"].copy()
    ep     = df[df["table"] == "episode"].copy()

    # Determine humans if not explicitly provided
    if human_labels is None:
        # infer labels from *_SA columns (e.g., H0_SA -> H0)
        human_labels = sorted({c[:-3] for c in steps.columns if c.endswith("_SA")})

    # Column lists
    sa_cols   = [f"{h}_SA"   for h in human_labels if f"{h}_SA"   in steps.columns]
    util_cols = [f"{h}_util" for h in human_labels if f"{h}_util" in steps.columns]

    # --- Mission completion time (MCT) inference ------------------------------
    def _infer_mct():
        mct_val = None
        try:
            if not ep.empty:
                # prefer mission_completion_time
                if "mission_completion_time" in ep.columns and pd.notna(ep["mission_completion_time"].iloc[-1]):
                    mct_val = float(ep["mission_completion_time"].iloc[-1])
                # fallback: episode_duration
                elif "episode_duration" in ep.columns and pd.notna(ep["episode_duration"].iloc[-1]):
                    mct_val = float(ep["episode_duration"].iloc[-1])
            # last resort: max time in step table
            if mct_val is None and not steps.empty and "t" in steps.columns:
                mct_val = float(steps["t"].max())
        except Exception as e:
            print(f"[viz] could not infer mission completion time: {e}")
        return mct_val

    mct = _infer_mct()

    def _annotate_mct(ax, mct_val):
        """Add a dashed vertical line at MCT and a right-middle label box."""
        if mct_val is None:
            return
        # ensure x-axis covers MCT
        x0, x1 = ax.get_xlim()
        if not np.isfinite(x0): x0 = 0.0
        if not np.isfinite(x1): x1 = mct_val
        if mct_val > x1:
            ax.set_xlim(x0, mct_val * 1.02)
        # vertical line + label box at right-middle
        ax.axvline(mct_val, linestyle="--", linewidth=1.6, alpha=0.6)
        ax.text(
            0.78, 0.5, f"MCT = {mct_val:.1f} s",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
            ha="left", va="center", fontsize=10
        )

    # 1) SA over time -----------------------------------------------------------
    if not steps.empty and sa_cols:
        fig1, ax1 = plt.subplots(figsize=(9, 4.6))
        for col in sa_cols:
            ax1.plot(steps["t"], steps[col], label=col[:-3], linewidth=2.0)
        ax1.set_title("Situation Awareness Score over Time")
        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("SA Score")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", ncol=min(3, len(sa_cols)))
        _annotate_mct(ax1, mct)
        fig1.tight_layout()

    # 2) Utilization over time --------------------------------------------------
    if not steps.empty and util_cols:
        fig2, ax2 = plt.subplots(figsize=(9, 4.6))
        for col in util_cols:
            ax2.plot(steps["t"], steps[col], label=col[:-5], linewidth=2.0)
        ax2.set_title("Human Utilization over Time (Sliding Window)")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Utilization [%]")
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", ncol=min(3, len(util_cols)))
        _annotate_mct(ax2, mct)
        fig2.tight_layout()

    # 3) Idle vs Overload ratios (episode aggregates) --------------------------
    if not agents.empty:
        names, idle_vals, ovl_vals = [], [], []
        for h in human_labels:
            row = agents[agents["agent"] == h]
            if not row.empty:
                idle = row["idle_ratio"].iloc[0]
                ovl  = row["overload_ratio"].iloc[0]
                idle = float(idle) * 100.0 if pd.notna(idle) else 0.0
                ovl  = float(ovl)  * 100.0 if pd.notna(ovl)  else 0.0
                names.append(h); idle_vals.append(idle); ovl_vals.append(ovl)

        if names:
            x = np.arange(len(names))
            width = 0.38
            fig3, ax3 = plt.subplots(figsize=(8.4, 4.8))
            bars1 = ax3.bar(x - width/2, idle_vals, width, label="Idle ratio")
            bars2 = ax3.bar(x + width/2, ovl_vals,  width, label="Overload ratio")
            ax3.set_title("Idle vs Overload Ratios (Episode)")
            ax3.set_ylabel("Ratio [%]")
            ax3.set_xticks(x, names)
            ax3.set_ylim(0, 100)
            ax3.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax3.grid(axis="y", alpha=0.3)
            ax3.legend(loc="best")

            # annotate bars
            for bars in (bars1, bars2):
                for b in bars:
                    val = b.get_height()
                    ax3.annotate(f"{val:.1f}%", xy=(b.get_x() + b.get_width()/2, val),
                                 xytext=(0, 3), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=9)
            fig3.tight_layout()

    if show:
        plt.show()


    def choose_metrics_csv(start_dir=None, title="Select unified metrics CSV"):
        """
        Open a native file picker to choose a CSV; returns the path or None.
        Falls back to a console selector if Tkinter is unavailable.
        """
        from pathlib import Path

        # Try a GUI file picker first
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                initialdir=str(start_dir) if start_dir else None,
                title=title,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            try:
                root.update()
            except Exception:
                pass
            try:
                root.destroy()
            except Exception:
                pass

            if path:
                return path
        except Exception as e:
            print(f"[viz] File dialog unavailable ({e}). Falling back to console selection.")

        # Fallback: list CSVs in a directory and ask for index in console
        if not start_dir:
            print("[viz] No start_dir provided and file dialog unavailable.")
            return None

        try:
            csvs = sorted(
                Path(start_dir).glob("*.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if not csvs:
                print(f"[viz] No CSV files found in: {start_dir}")
                return None

            print("\n[viz] Select a CSV by index:")
            for i, p in enumerate(csvs[:50]):
                print(f"  [{i:2d}] {p.name}")
            sel = input("Index (ENTER to cancel): ").strip()
            if sel == "":
                return None
            idx = int(sel)
            return str(csvs[idx])
        except Exception as e:
            print(f"[viz] Console selection failed: {e}")
            return None
# ------------------------------------------------------------------------------

def animate_workspace(
    ws,
    steps=200,
    interval=100,
    sim=None,
    dt=0.1,
    record=False,
    output_path="workspace.gif",
):
    """
    Animate the workspace, reflected across the line y = x (visual swap of coordinates).

    Modes:
      • Live mode: provide `sim`; calls sim.step(dt=..., mode="sim", verbose=False) each frame.
      • Playback mode: if `sim` is None, replays recorded agent.traj / progress_traj.

    Args:
        ws: Workspace
        steps: number of frames to animate (cap for live mode; full length for playback)
        interval: delay between frames in ms
        sim: Simulation or None
        dt: timestep used in live mode when calling sim.step()
        record: if True, saves a GIF to output_path (Pillow writer)
        output_path: path to .gif
    """
    # local import so you don't have to edit module-level imports
    from ltl_core import env_ltl as env

    fig, ax = plt.subplots(figsize=(8, 8))

    # Because we flip across y=x, swap the axis extents too:
    ax.set_xlim(0, ws.size[0])   # was ws.size[1]
    ax.set_ylim(0, ws.size[1])   # was ws.size[0]
    ax.set_aspect('equal')
    ax.grid(True)

    # --- Static areas (swap x,y for visual flip) ---
    for (x, y) in ws.base_area:
        ax.add_patch(
            patches.Rectangle((x, y), 1, 1, linewidth=1,
                              edgecolor='blue', facecolor='lightblue', alpha=0.3, zorder=1)
        )
    for (x, y) in ws.hospital_area:
        ax.add_patch(
            patches.Rectangle((x, y), 1, 1, linewidth=1,
                              edgecolor='red', facecolor='mistyrose', alpha=0.3, zorder=1)
        )

    # --- Obstacles (draw beneath targets/agents) ---
    obstacle_artists = []
    # Rectangular obstacles: boundary + internal rectangles
    for (x, y, w, h) in env.Env().obs_boundary + env.Env().obs_rectangle:
        r = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.0,
            edgecolor=(0.2, 0.2, 0.2, 1.0),
            facecolor=(0.5, 0.5, 0.5, 0.55),
            zorder=2
        )
        ax.add_patch(r)
        obstacle_artists.append(r)
    # Circular obstacles
    for (cx, cy, rad) in env.Env().obs_circle:
        c = plt.Circle(
            (cx, cy), rad,
            linewidth=1.0,
            edgecolor=(0.25, 0.05, 0.05, 1.0),
            facecolor=(0.7, 0.1, 0.1, 0.5),
            zorder=2
        )
        ax.add_patch(c)
        obstacle_artists.append(c)

    # --- Targets (swap x,y for visual flip) ---
    target_dots = []
    target_texts = []
    for i, (x, y) in enumerate(ws.target_locations):
        dot, = ax.plot(x + 0.5, y + 0.5, 'ro', markersize=10, zorder=4)
        txt = ax.text(x + 0.5, y + 0.5, f"T{i}", color='black',
                      fontsize=12, ha='center', va='center', zorder=5)
        target_dots.append(dot)
        target_texts.append(txt)

    # --- Mobile agents (drones, GVs) ---
    agent_dots = {}
    for role in ["drones", "gvs"]:
        color = {'drones': 'b', 'gvs': 'g'}[role]
        for agent in ws.agents[role]:
            dot, = ax.plot([], [], 'o', label=agent.label, color=color, markersize=10, zorder=6)
            agent_dots[agent.label] = dot

    # --- Humans (static positions) ---
    for agent in ws.agents["humans"]:
        x, y = agent.pos
        ax.plot(x + 0.5, y + 0.5, 'o', color='orange', markersize=10, zorder=6)
        ax.text(x + 0.5, y + 0.5, agent.label, fontsize=12, ha='center', va='center', zorder=7)

    # --- Per-frame overlays ---
    progress_halos = []
    progress_texts = []
    # Put step text near the (flipped) top-left corner
    step_text = ax.text(0.5, ws.size[1] - 0.5, "", fontsize=12, color='gray', ha='left', zorder=8)

    live_mode = sim is not None

    def init():
        for dot in agent_dots.values():
            dot.set_data([], [])
        step_text.set_text("")
        # return static obstacle artists too (even though blit=False, keeps references tidy)
        return obstacle_artists + list(agent_dots.values()) + target_dots + target_texts + [step_text]

    def update(frame):
        # Clear previous overlays
        for halo in progress_halos:
            halo.remove()
        for txt in progress_texts:
            txt.remove()
        progress_halos.clear()
        progress_texts.clear()

        # Advance simulation in live mode
        if live_mode:
            sim.step(dt=dt, mode="sim", verbose=False)
            step_text.set_text(f"t = {frame * dt:.1f}s")
        else:
            step_text.set_text(f"Step: {frame}")

        # --- Drones & GVs ---
        for role in ["drones", "gvs"]:
            for agent in ws.agents[role]:
                # Source position: live = current pos; playback = traj[frame]
                if live_mode:
                    row, col = agent.pos
                else:
                    if frame >= len(agent.traj):
                        continue
                    row, col = agent.traj[frame]

                # Flip across y=x by swapping x/y for plotting
                x_plot, y_plot = row + 0.5, col + 0.5

                # set_data needs sequences
                agent_dots[agent.label].set_data([x_plot], [y_plot])

                # Progress halo & text
                if live_mode:
                    task = getattr(agent, "current_symbolic_task", None)
                    p_val = agent.get_progress(task) if task else 0.0
                else:
                    p_val = agent.progress_traj[frame] if frame < len(agent.progress_traj) else 0.0

                if p_val > 0:
                    halo, = ax.plot([x_plot], [y_plot], 'o', color='lime', markersize=15, alpha=p_val * 0.6, zorder=7)
                    text = ax.text(x_plot, y_plot + 0.3, f"{p_val:.1f}", fontsize=8, color='green', ha='center', zorder=8)
                    progress_halos.append(halo)
                    progress_texts.append(text)

        # --- Humans: progress halo (position static but show progress if any) ---
        for agent in ws.agents["humans"]:
            row, col = agent.pos
            x_plot, y_plot = row + 0.5, col + 0.5  # flipped
            if live_mode:
                task = getattr(agent, "current_symbolic_task", None)
                p_val = agent.get_progress(task) if task else 0.0
            else:
                p_val = agent.progress_traj[frame] if frame < len(agent.progress_traj) else 0.0

            if p_val > 0:
                halo, = ax.plot([x_plot], [y_plot], 'o', color='lime', markersize=15, alpha=p_val * 0.6, zorder=7)
                text = ax.text(x_plot, y_plot + 0.3, f"{p_val:.1f}", fontsize=8, color='green', ha='center', zorder=8)
                progress_halos.append(halo)
                progress_texts.append(text)

        return obstacle_artists + list(agent_dots.values()) + progress_halos + progress_texts + target_dots + target_texts + [step_text]

    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=steps, interval=interval, blit=False
    )

    plt.legend()
    plt.tight_layout()

    if record:
        ani.save(output_path, writer="pillow")

    return ani


def _ap_prefix(ap: str) -> str:
    if not isinstance(ap, str) or not ap:
        return "idle"
    try:
        pref = get_ap_prefix(ap)  # e.g., "p_scan"
        if pref.startswith("p_"):
            return pref if pref in AP_COLORS else pref  # keep unknown tasks explicit
        return "env"
    except Exception:
        return "env"


class GuiEpisodeLogger:
    """
    Collects GUI-side time series and builds:
      - gui_episode_<timestamp>_unified.csv
      - gui_episode_<timestamp>_1080p.png (1920x1080)

    Minimal API:
      logger = GuiEpisodeLogger(log_dir, agents, human_labels=None)
      logger.note_step(t, assignments, completed, humans)
      csv_path, png_path = logger.save_all(mct=mission_completed_time)
    """
    def __init__(self, log_dir: Path, agents, human_labels=None,
                 fig_size_inch=(19.2, 10.8), csv_prefix="gui_episode"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # agent order (labels)
        self.agent_labels = []
        for a in list(agents):
            lab = getattr(a, "label", None)
            if lab:
                self.agent_labels.append(lab)
        # humans
        if human_labels is None:
            self.human_labels = [lab for lab in self.agent_labels if lab.startswith("H")]
        else:
            self.human_labels = list(human_labels)

        # time series
        self.times = []
        self.assigned_cnt = []
        self.completed_cnt = []
        self.sa_by_h = {h: [] for h in self.human_labels}
        self.util_by_h = {h: [] for h in self.human_labels}

        # timeline segments
        self._cur_seg = {}                 # lab -> (t0, prefix)
        self.segments_by_agent = {}        # lab -> [(t0, t1, prefix), ...]

        # fig config
        self.fig_size_inch = fig_size_inch
        self.csv_prefix = csv_prefix

    def note_step(self, t, assignments: dict, completed, humans):
        """
        assignments: dict {AgentObj -> ap_str}  (None if idle)
        completed  : list/iterable of APs completed this tick (for count only)
        humans     : iterable of Human Agent objects (with .label, .sa_score, .utilization)
        """
        self.times.append(float(t))
        # counts
        if isinstance(assignments, dict):
            self.assigned_cnt.append(sum(1 for ap in assignments.values() if ap))
        else:
            self.assigned_cnt.append(0)
        self.completed_cnt.append(len(completed) if completed is not None else 0)
        # human metrics
        for h in humans:
            lab = getattr(h, "label", None)
            if lab in self.sa_by_h:
                self.sa_by_h[lab].append(float(getattr(h, "sa_score", 0.0)))
                self.util_by_h[lab].append(float(getattr(h, "utilization", 0.0)))

        # timeline: map label -> ap
        label2ap = {}
        if isinstance(assignments, dict):
            for agent_obj, ap in assignments.items():
                lab = getattr(agent_obj, "label", None)
                if lab:
                    label2ap[lab] = ap

        for lab in self.agent_labels:
            pref = _ap_prefix(label2ap.get(lab))
            tseg = self._cur_seg.get(lab)
            if tseg is None:
                self._cur_seg[lab] = (float(t), pref)
            else:
                t0, prev_pref = tseg
                if pref != prev_pref:
                    self.segments_by_agent.setdefault(lab, []).append((t0, float(t), prev_pref))
                    self._cur_seg[lab] = (float(t), pref)

    def _close_segments(self):
        final_t = self.times[-1] if self.times else 0.0
        for lab, (t0, pref) in list(self._cur_seg.items()):
            self.segments_by_agent.setdefault(lab, []).append((t0, final_t, pref))

    def _write_unified_csv(self, csv_path: Path, mct=None):
        step_cols = ["table", "t"] \
                    + [f"{h}_SA" for h in self.human_labels] \
                    + [f"{h}_util" for h in self.human_labels] \
                    + ["assigned_cnt", "completed_cnt"]
        seg_cols  = ["table", "agent", "role", "prefix", "t_start", "t_end"]
        epi_cols  = ["table", "mission_completion_time"]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            # step header
            w.writerow(step_cols)
            for i, t in enumerate(self.times):
                row = ["step", t]
                for h in self.human_labels:
                    row.append(self.sa_by_h[h][i] if i < len(self.sa_by_h[h]) else "")
                for h in self.human_labels:
                    row.append(self.util_by_h[h][i] if i < len(self.util_by_h[h]) else "")
                row += [self.assigned_cnt[i], self.completed_cnt[i]]
                w.writerow(row)

            # segment header + rows
            w.writerow(seg_cols)
            for lab in sorted(self.segments_by_agent.keys()):
                role = "drone" if lab.startswith("D") else ("gv" if lab.startswith("G") else ("human" if lab.startswith("H") else ""))
                for (t0, t1, pref) in self.segments_by_agent[lab]:
                    w.writerow(["segment", lab, role, pref, t0, t1])

            # episode row
            w.writerow(epi_cols)
            w.writerow(["episode", mct if mct is not None else ""])

    def _plot_1080p(self, png_path, mct=None, show=False):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.ticker import PercentFormatter
        times = self.times

        fig = plt.figure(figsize=self.fig_size_inch, dpi=100)
        gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2], hspace=0.28, wspace=0.2)

        # 1) SA
        ax1 = fig.add_subplot(gs[0, 0])
        for h in self.human_labels:
            if self.sa_by_h[h]:
                ax1.plot(times, self.sa_by_h[h], linewidth=2, label=h)
        ax1.set_title("Situation Awareness over Time"); ax1.set_xlabel("Time [s]"); ax1.set_ylabel("SA")
        ax1.grid(True, alpha=0.3); 
        if self.human_labels: ax1.legend(loc="best")
        if mct is not None:
            ax1.axvline(mct, linestyle="--", linewidth=1.6, alpha=0.6)
            ax1.text(0.78, 0.5, f"MCT = {mct:.1f} s", transform=ax1.transAxes,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
                     ha="left", va="center", fontsize=10)

        # 2) Util
        ax2 = fig.add_subplot(gs[0, 1])
        for h in self.human_labels:
            if self.util_by_h[h]:
                ax2.plot(times, self.util_by_h[h], linewidth=2, label=h)
        ax2.set_title("Human Utilization over Time (Sliding Window)")
        ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Utilization [%]")
        ax2.set_ylim(0, 100); ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax2.grid(True, alpha=0.3); 
        if self.human_labels: ax2.legend(loc="best")
        if mct is not None:
            ax2.axvline(mct, linestyle="--", linewidth=1.6, alpha=0.6)
            ax2.text(0.78, 0.5, f"MCT = {mct:.1f} s", transform=ax2.transAxes,
                     bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
                     ha="left", va="center", fontsize=10)

        # 3) Counts
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(times, self.assigned_cnt, linewidth=2, label="assigned_cnt")
        ax3.plot(times, self.completed_cnt, linewidth=2, label="completed_cnt")
        ax3.set_title("Assignments and Completions per Tick")
        ax3.set_xlabel("Time [s]"); ax3.set_ylabel("Count"); ax3.grid(True, alpha=0.3); ax3.legend(loc="best")
        if mct is not None:
            ax3.axvline(mct, linestyle="--", linewidth=1.6, alpha=0.6)

        # 4) Timeline
        ax4 = fig.add_subplot(gs[2, :])
        # order D*, G*, H* top→bottom
        ordered = sorted([l for l in self.agent_labels if l.startswith("D")]) \
                + sorted([l for l in self.agent_labels if l.startswith("G")]) \
                + sorted([l for l in self.agent_labels if l.startswith("H")])
        y_pos = {lab: (len(ordered)-1-idx) for idx, lab in enumerate(ordered)}
        for lab in ordered:
            for (t0, t1, pref) in self.segments_by_agent.get(lab, []):
                if t1 is None or t1 < t0: 
                    continue
                color = AP_COLORS.get(pref, "#5a5a5a")
                ax4.hlines(y=y_pos[lab], xmin=t0, xmax=t1, colors=color, linewidth=8, alpha=0.9)
                if (t1 - t0) >= 3.0:
                    xc = 0.5*(t0 + t1)
                    ax4.text(xc, y_pos[lab]+0.18, AP_ABBR.get(pref, pref[2:].upper()),
                             ha="center", va="bottom", fontsize=8)
        ax4.set_title("Agent Task Timeline"); ax4.set_xlabel("Time [s]")
        ax4.set_yticks([y_pos[l] for l in ordered]); ax4.set_yticklabels(ordered)
        ax4.grid(True, axis="x", alpha=0.25); ax4.set_ylim(-1, len(ordered))
        if mct is not None:
            ax4.axvline(mct, linestyle="--", linewidth=1.6, alpha=0.6)

        fig.suptitle("GUI Episode Summary", fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(png_path, dpi=100)
        if show:
            plt.show()

    def save_all(self, mct=None, show=False):
        from time import strftime
        self._close_segments()
        stamp = strftime("%Y%m%d_%H%M%S")
        csv_path = self.log_dir / f"{self.csv_prefix}_{stamp}_unified.csv"
        png_path = self.log_dir / f"{self.csv_prefix}_{stamp}_1080p.png"
        self._write_unified_csv(csv_path, mct=mct)
        self._plot_1080p(png_path, mct=mct, show=show)
        return str(csv_path), str(png_path)