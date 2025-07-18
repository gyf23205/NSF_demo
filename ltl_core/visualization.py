import re
import pygame
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph


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
    kind = parts[1]
    region = parts[2] if len(parts) > 2 else ''
    m = {
      'search':       'Search',
      'rescue':       'Rescue',
      'oversight':    'Oversight',
      'navscan':      'NavScan',
      'confirm':      'Confirm',
      'foundgate':    'Found',
      'notfoundgate': 'Not Found',
      'skip':         'Skip',
      'pickup':       'Pick Up',
      'dropoff':      'Drop Off',
      'monitor':      'Monitor',
      'rank':         'Rank',
      'submit':       'Report',
    }
    label = m.get(kind, node)
    return f"{label} {region}".strip()


def draw_composite_hierarchy(G: nx.DiGraph, spec, figsize=(10, 6)):
    """
    - Lock Level-3 (generalised functions) on one horizontal rank
      and lay them left→right in the order [search, rescue, skip, …]
      with *uniform* spacing.
    - Level-4 nodes keep Graphviz's vertical placement but take the
      x-position of their Level-3 parent so they stay centred.
    - Everything else (styling, centring of Level-1 / Level-2) unchanged.
    """
    # 1) Composite-only sub-graph
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
            # 🔸 All children are connected: vertical stack at parent's x
            x = pos[parent][0]
            sorted_chain = sorted(components[0], key=lambda n: pos[n][1])
            y_start = min(pos[n][1] for n in sorted_chain)

            for i, node in enumerate(sorted_chain):
                pos[node] = (x, y_start + i * 40)
        else:
            # 🔹 Multiple disconnected chains: spread horizontally around parent
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
        "Generalised Function",
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


def animate_workspace(ws, steps=20, interval=500):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, ws.size[1])
    ax.set_ylim(0, ws.size[0])
    ax.set_aspect('equal')
    ax.grid(True)

    for (x, y) in ws.base_area:
        ax.add_patch(patches.Rectangle((y, x), 1, 1, linewidth=1,
                                       edgecolor='blue', facecolor='lightblue', alpha=0.3))
    for (x, y) in ws.hospital_area:
        ax.add_patch(patches.Rectangle((y, x), 1, 1, linewidth=1,
                                       edgecolor='red', facecolor='mistyrose', alpha=0.3))

    target_dots = []
    target_texts = []
    for i, (x, y) in enumerate(ws.target_locations):
        dot, = ax.plot(y + 0.5, x + 0.5, 'ro', markersize=10)
        txt = ax.text(y + 0.5, x + 0.5, f"T{i}", color='black',
                      fontsize=12, ha='center', va='center')
        target_dots.append(dot)
        target_texts.append(txt)

    agent_dots = {}
    for role in ["drones", "gvs"]:
        for agent in ws.agents[role]:
            dot, = ax.plot([], [], 'o', label=agent.label,
                           color={'drones': 'b', 'gvs': 'g'}[role], markersize=10)
            agent_dots[agent.label] = dot

    for agent in ws.agents["humans"]:
        x, y = agent.pos
        ax.plot(y + 0.5, x + 0.5, 'o', color='orange', markersize=10)
        ax.text(y + 0.5, x + 0.5, agent.label, fontsize=12, ha='center', va='center')

    progress_halos = []
    progress_texts = []
    step_text = ax.text(0.5, ws.size[0] - 0.5, "", fontsize=12, color='gray', ha='left')

    def init():
        for dot in agent_dots.values():
            dot.set_data([], [])
        step_text.set_text("")
        return list(agent_dots.values()) + target_dots + target_texts + [step_text]

    def update(frame):
        for halo in progress_halos:
            halo.remove()
        for txt in progress_texts:
            txt.remove()
        progress_halos.clear()
        progress_texts.clear()

        step_text.set_text(f"Step: {frame}")

        for role in ["drones", "gvs"]:
            for agent in ws.agents[role]:
                if frame < len(agent.traj):
                    row, col = agent.traj[frame]
                    x_plot, y_plot = col + 0.5, row + 0.5
                    agent_dots[agent.label].set_data(x_plot, y_plot)

                    p_val = agent.progress_traj[frame] if frame < len(agent.progress_traj) else 0.0
                    if p_val > 0:
                        halo, = ax.plot(x_plot, y_plot, 'o', color='lime', markersize=15, alpha=p_val * 0.6)
                        text = ax.text(x_plot, y_plot + 0.3, f"{p_val:.1f}", fontsize=8, color='green', ha='center')
                        progress_halos.append(halo)
                        progress_texts.append(text)

        for agent in ws.agents["humans"]:
            row, col = agent.pos
            x_plot, y_plot = col + 0.5, row + 0.5
            p_val = agent.progress_traj[frame] if frame < len(agent.progress_traj) else 0.0
            if p_val > 0:
                halo, = ax.plot(x_plot, y_plot, 'o', color='lime', markersize=15, alpha=p_val * 0.6)
                text = ax.text(x_plot, y_plot + 0.3, f"{p_val:.1f}", fontsize=8, color='green', ha='center')
                progress_halos.append(halo)
                progress_texts.append(text)

        return list(agent_dots.values()) + progress_halos + progress_texts + target_dots + target_texts + [step_text]

    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=steps, interval=interval, blit=False)
    plt.legend()
    plt.tight_layout()
    return ani
