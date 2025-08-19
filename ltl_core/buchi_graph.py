import os
import sys
import re
import subprocess
import shutil
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
from pathlib import Path

TRANSLATOR_USED = None  # optional: for debugging/verification

# Updated for Linux
def run_ltl2ba(ltl_formula: str, ltl2ba_path: str | None = None) -> str:
    """
    Use *ltl2ba* only (no Spot fallback).
    Preference:
      Windows:   <repo>/ltl_core/ltl2ba.exe -> PATH ltl2ba.exe/ltl2ba -> explicit arg -> $LTL2BA
      Linux/mac: <repo>/ltl_core/ltl2ba     -> PATH ltl2ba           -> explicit arg -> $LTL2BA
    On non-Windows, we NEVER try a .exe.
    """
    here = Path(__file__).parent
    is_windows = (os.name == "nt") or sys.platform.startswith("win")

    # Normalize a few unicode operators to ltl2ba syntax
    cleaned = (
        ltl_formula
        .replace("∧", "&&")
        .replace("∨", "||")
        .replace("→", "->")
        .replace("¬", "!")
    )

    candidates: list[str] = []

    # 1) Repo-local binary first (your working Linux binary lives here)
    if is_windows:
        local_win = here / "ltl2ba.exe"
        if local_win.exists():
            candidates.append(str(local_win))
    else:
        local_lin = here / "ltl2ba"
        if local_lin.exists():
            # ensure it's executable
            try:
                local_lin.chmod(local_lin.stat().st_mode | 0o111)
            except Exception:
                pass
            candidates.append(str(local_lin))

    # 2) PATH
    if is_windows:
        for name in ("ltl2ba.exe", "ltl2ba"):
            w = shutil.which(name)
            if w:
                candidates.append(w)
    else:
        w = shutil.which("ltl2ba")
        if w:
            candidates.append(w)

    # 3) Explicit function argument
    if ltl2ba_path:
        if os.path.isfile(ltl2ba_path):
            candidates.append(ltl2ba_path)
        else:
            w = shutil.which(ltl2ba_path)
            if w:
                candidates.append(w)

    # 4) Environment override (last)
    env_override = os.getenv("LTL2BA")
    if env_override:
        if os.path.isfile(env_override):
            candidates.append(env_override)
        else:
            w = shutil.which(env_override)
            if w:
                candidates.append(w)

    # De-dup while preserving order
    seen, uniq = set(), []
    for c in candidates:
        if c and c not in seen:
            uniq.append(c); seen.add(c)

    last_err = None
    for cmd in uniq:
        # never try a Windows .exe on non-Windows
        if (not is_windows) and cmd.lower().endswith(".exe"):
            continue
        try:
            res = subprocess.run([cmd, "-f", cleaned], capture_output=True, text=True, check=True)
            globals()["TRANSLATOR_USED"] = cmd
            return res.stdout
        except (subprocess.CalledProcessError, PermissionError, OSError) as e:
            # try to add +x if it's a file we control, then move on
            try:
                p = Path(cmd)
                if p.exists():
                    p.chmod(p.stat().st_mode | 0o111)
            except Exception:
                pass
            last_err = e
            continue

    raise FileNotFoundError(
        "ltl2ba not found/usable. Provide a native ltl2ba for this OS.\n"
        "Tried (in order): <repo>/ltl_core/ltl2ba(.exe), PATH, explicit arg, $LTL2BA.\n"
        f"Last error: {last_err}"
    )


# For Windows only; legacy code kept for reference
# def run_ltl2ba(
#     ltl_formula: str,
#     ltl2ba_path: str = "ltl2ba.exe"
# ) -> str:
#     """
#     Invoke the **nondeterministic** ltl2ba executable to generate a Büchi automaton.
#     Replaces Unicode ∧/∨ with &&/|| so the formula parses correctly.
#     Raises FileNotFoundError if the executable cannot be found.
#     """
#     # Try to resolve local path first
#     local_path = os.path.join(os.path.dirname(__file__), "ltl2ba.exe")
#     if os.path.isfile(local_path):
#         path = local_path
#     elif os.path.isfile(ltl2ba_path):
#         path = ltl2ba_path
#     elif shutil.which(ltl2ba_path) is not None:
#         path = shutil.which(ltl2ba_path)
#     else:
#         raise FileNotFoundError(
#             f"ltl2ba executable not found (tried local 'ltl2ba.exe' and PATH)"
#         )

#     # clean formula
#     cleaned = ltl_formula.replace("∧", "&&").replace("∨", "||")

#     # run without '-d' to keep it nondeterministic
#     result = subprocess.run(
#         [path, "-f", cleaned],
#         capture_output=True,
#         text=True,
#         check=True
#     )
#     return result.stdout


def parse_ltl2ba_output(ba_raw: str) -> nx.MultiDiGraph:
    """
    Parse SPIN-style NBA (with -d) into a MultiDiGraph.
    • State headers: any line ending in ':' that isn’t 'Formula:' or 'if'/'fi;'
    • Transitions: lines starting with '::' and ending '-> goto STATE'
    • Prune any guard containing '&&', the unconditional '1', or self-loops
    • Split disjunctions '(A) || (B)' into two edges labeled A and B
    • Keep plain single-AP guards '(A)' as edge A
    """
    G = nx.MultiDiGraph()
    current = None

    # transition regex: capture everything inside the outermost parentheses
    trans_re = re.compile(r'^::\s*\((.+)\)\s*->\s*(?:goto\s+)?([A-Za-z0-9_]+)')

    for line in ba_raw.splitlines():
        text = line.strip()
        if not text:
            continue

        # 1) State header (skip 'Formula:', 'if', 'fi;')
        if text.endswith(':') and not text.lower().startswith(('formula:', 'if', 'fi')):
            state = text[:-1].strip()
            current = state
            G.add_node(current)
            continue

        # 2) Transition line
        m = trans_re.match(text)
        if not m or current is None:
            continue

        guard, tgt = m.group(1).strip(), m.group(2).strip()

        # 3) Prune conjunctions, unconditional loops, self-loops
        if '&&' in guard or guard in ('1', '(1)') or tgt == current:
            continue

        # 4) Split binary disjunctions "(A) || (B)"
        if '||' in guard:
            parts = re.split(r'\)\s*\|\|\s*\(', guard)
            for part in parts:
                ap = part.strip('() ')
                if re.fullmatch(r'p_[A-Za-z0-9_]+', ap):
                    G.add_edge(current, tgt, label={ap})
            continue

        # 5) Keep plain single-AP guards "(A)"
        ap = guard.strip('() ')
        if re.fullmatch(r'p_[A-Za-z0-9_]+', ap):
            G.add_edge(current, tgt, label={ap})

    return G


def ltl_to_nba(
    ltl_formula: str,
    ltl2ba_path: str = "ltl2ba.exe"
) -> nx.MultiDiGraph:
    """
    Convenience wrapper: runs ltl2ba then parses the output.
    """
    raw = run_ltl2ba(ltl_formula, ltl2ba_path)
    return parse_ltl2ba_output(raw)


def print_automaton_graph(
    G: nx.MultiDiGraph,
    title: str = "Büchi Automaton"
) -> None:
    """
    Print each transition of the automaton in a readable form.
    """
    print(f"\n=== {title} ===")
    for u, v, data in G.edges(data=True):
        print(f"{u} → {v} [ {data['label']} ]")


def plot_automaton_graph(
    G: nx.MultiDiGraph,
    title: str = "Büchi Automaton"
) -> None:
    """
    Plot the automaton using matplotlib.
    Accepting states (containing 'accept' in their name) are highlighted.
    """
    pos = nx.spring_layout(G, seed=42)
    accepting = [n for n in G.nodes if "accept" in n.lower()]
    normal = [n for n in G.nodes if n not in accepting]

    nx.draw_networkx_nodes(G, pos,
                           nodelist=normal,
                           node_color="skyblue",
                           node_size=800)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=accepting,
                           node_color="lightgreen",
                           node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # for a MultiDiGraph, edges(data=True) yields one entry per key
    edge_labels = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        edge_labels[(u, v, key)] = data["label"]

    nx.draw_networkx_edges(G, pos, arrows=True)
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=edge_labels,
                                 font_size=7)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
