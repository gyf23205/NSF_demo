import re
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict
from networkx.drawing.nx_agraph import graphviz_layout

from .buchi_graph import run_ltl2ba, parse_ltl2ba_output


# --------------------------------------------------------------------------- #
# DAG construction helpers
# --------------------------------------------------------------------------- #
_AP_RE: re.Pattern = re.compile(r"p_[A-Za-z0-9_]+")


def _extract_alt_groups(formula: str) -> List[List[str]]:
    """
    Return lists of alternatives that appear inside the *same* parenthesis with
    a logical OR.

    Example
    -------
    "(p_rescue_0 || p_skip_0)"          ->  [["p_rescue_0", "p_skip_0"]]
    "(a && (b || c) && (d || e || f))" ->  [["b", "c"], ["d", "e", "f"]]
    """
    groups: List[List[str]] = []
    # find every "( ... || ... )" without nesting-aware parsing (sufficient here)
    for m in re.finditer(r"\(([^()]*\|\|[^()]*)\)", formula):
        inside = m.group(1)
        group_aps = _AP_RE.findall(inside)
        if len(group_aps) >= 2:
            groups.append(group_aps)
    return groups


def _all_children(formula: str) -> List[str]:
    """Return every AP / composite token referenced in *formula*."""
    return _AP_RE.findall(formula)


def build_dag(specification) -> nx.DiGraph:
    composite_names, atomic_names = specification.get_all_function_names()
    G = nx.DiGraph()

    G.graph["composite_names"] = composite_names
    G.graph["atomic_names"] = atomic_names

    pairwise_preds: Dict[str, List[str]] = defaultdict(list)

    for n in composite_names + atomic_names:
        G.add_node(n)

    for lvl, level in enumerate(specification.hierarchy, start=1):
        print(f"\n--- Level {lvl} ---")
        for parent, formula in level.items():
            print(f"    ▶ {parent}: {formula}")
            children = _all_children(formula)

            # Manually add auxiliary children if formula is empty but pattern matches
            # if not children and parent.startswith("p_101"):
            #     children = [name for name in level if name.startswith(f"{parent}_aux_")]

            # print(f"    → children of {parent}: {children}")
            for child in children:
                if (child in composite_names or child in atomic_names) and not G.has_edge(parent, child):
                    G.add_edge(parent, child)
                    print(f"parent→child:    {parent} → {child}")

            if formula.strip():
                for a, b in extract_pairwise_order_ltl2ba(formula):
                    if a in G and b in G and not G.has_edge(a, b):
                        G.add_edge(a, b)
                        print(f"pairwise order:  {parent}: {a} → {b}")
                    pairwise_preds[b].append(a)

            alt_groups = _extract_alt_groups(formula)
            in_alts = {ap for grp in alt_groups for ap in grp}
            mandatory = [c for c in children if c not in in_alts]

            if alt_groups or mandatory:
                G.nodes[parent]["alt_groups"] = alt_groups
                G.nodes[parent]["mandatory"] = mandatory

    G.graph["predecessors_map"] = {n: [u for u, _ in G.in_edges(n)] for n in G.nodes}
    G.graph["pairwise_predecessors"] = dict(pairwise_preds)

    return G


# --------------------------------------------------------------------------- #
# ltl2ba-based ordering extraction (unchanged except refactor)
# --------------------------------------------------------------------------- #
def extract_pairwise_order_ltl2ba(formula: str) -> List[Tuple[str, str]]:
    raw = run_ltl2ba(formula)
    nba = parse_ltl2ba_output(raw)  # MultiDiGraph
    init = next(iter(nba.nodes), None)
    accepting = {n for n in nba if "accept" in n.lower()}
    pairs: List[Tuple[str, str]] = []

    # ---------- strict ordering via path analysis ----------
    if init and accepting:
        traces: List[List[str]] = []

        def dfs(u: str, trace: List[str]):
            if u in accepting:
                traces.append(trace)
                return
            for _, v, d in nba.out_edges(u, data=True):
                dfs(v, trace + [d["label"]])

        dfs(init, [])

        before, co = defaultdict(int), defaultdict(int)
        for tr in traces:
            unique = set(tr)
            for a in unique:
                for b in unique:
                    if a != b:
                        co[(a, b)] += 1
            for i, a in enumerate(tr):
                for b in tr[i + 1 :]:
                    before[(a, b)] += 1

        for (a, b), cnt in before.items():
            if cnt == co[(a, b)] > 0 and before.get((b, a), 0) == 0:
                pairs.append((a, b))

    # ---------- simple regex fallback for OR-formulas ----------
    if not pairs and ("||" in formula or "∨" in formula):
        for part in re.split(r"\|\||∨", formula):
            m = re.search(r"(p_[A-Za-z0-9_]+)\s*&&\s*<>\s*(p_[A-Za-z0-9_]+)", part)
            if m:
                pairs.append((m.group(1), m.group(2)))

    return pairs


# --------------------------------------------------------------------------- #
# Optional drawing utilities (unchanged)
# --------------------------------------------------------------------------- #
def draw_composite_propositions(G: nx.DiGraph, figsize=(8, 6)) -> None:
    composite_names = G.graph.get("composite_names", [])
    H = G.subgraph(composite_names).copy()
    pos = graphviz_layout(H, prog="dot")
    plt.figure(figsize=figsize)
    nx.draw(
        H, pos,
        with_labels=True,
        node_size=600,
        arrowsize=12,
        font_size=10
    )
    plt.title("Composite Propositions (Parent→Child & Pairwise)")
    plt.axis("off")
    plt.show()


def draw_atomic_propositions(G: nx.DiGraph, figsize=(8, 6)) -> None:
    atomic_names = G.graph.get("atomic_names", [])
    H = G.subgraph(atomic_names).copy()
    pos = graphviz_layout(H)
    plt.figure(figsize=figsize)
    nx.draw(
        H, pos,
        with_labels=True,
        node_color="lightcoral",
        node_size=500,
        arrowsize=12,
        font_size=10
    )
    plt.title("Atomic Propositions (Pairwise)")
    plt.axis("off")
    plt.show()
