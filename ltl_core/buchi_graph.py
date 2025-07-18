import os
import re
import subprocess
import shutil
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union


def run_ltl2ba(
    ltl_formula: str,
    ltl2ba_path: str = "ltl2ba.exe"
) -> str:
    """
    Invoke the **nondeterministic** ltl2ba executable to generate a Büchi automaton.
    Replaces Unicode ∧/∨ with &&/|| so the formula parses correctly.
    Raises FileNotFoundError if the executable cannot be found.
    """
    # Try to resolve local path first
    local_path = os.path.join(os.path.dirname(__file__), "ltl2ba.exe")
    if os.path.isfile(local_path):
        path = local_path
    elif os.path.isfile(ltl2ba_path):
        path = ltl2ba_path
    elif shutil.which(ltl2ba_path) is not None:
        path = shutil.which(ltl2ba_path)
    else:
        raise FileNotFoundError(
            f"ltl2ba executable not found (tried local 'ltl2ba.exe' and PATH)"
        )

    # clean formula
    cleaned = ltl_formula.replace("∧", "&&").replace("∨", "||")

    # run without '-d' to keep it nondeterministic
    result = subprocess.run(
        [path, "-f", cleaned],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


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
                    G.add_edge(current, tgt, label=ap)
            continue

        # 5) Keep plain single-AP guards "(A)"
        ap = guard.strip('() ')
        if re.fullmatch(r'p_[A-Za-z0-9_]+', ap):
            G.add_edge(current, tgt, label=ap)

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
