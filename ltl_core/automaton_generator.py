from .buchi_graph import run_ltl2ba, parse_ltl2ba_output
import networkx as nx
from typing import Dict, Any, Set, List
from collections import defaultdict


def compile_automata(spec) -> Dict[str, nx.DiGraph]:
    """
    Compile each LTL sub-formula in spec.hierarchy into a DFA represented
    as a NetworkX graph, with attached initial_state, accepting_states, and delta().

    Returns:
        automata: dict mapping formula-name -> NetworkX DiGraph (the DFA)
    """
    automata: Dict[str, nx.DiGraph] = {}

    for level in spec.hierarchy:
        for name, formula in level.items():
            if not formula.strip():
                continue  # skip structural/auxiliary node like p_101

            # 1) parse the Buchi automaton into a directed graph
            raw = run_ltl2ba(formula)
            G: nx.DiGraph = parse_ltl2ba_output(raw)

            # 2) identify initial state (node with in-degree zero or flagged)
            in_degrees = dict(G.in_degree())
            init_candidates = [n for n, deg in in_degrees.items() if deg == 0]
            # include node-level 'initial' flags
            for n, data in G.nodes(data=True):
                if data.get('initial', False) and n not in init_candidates:
                    init_candidates.append(n)
            if not init_candidates:
                raise RuntimeError(f"No initial state found for automaton '{name}'")
            q0 = init_candidates[0]

            # 3) identify accepting states from any source
            accepting: Set[Any] = set()
            # 3a) graph-level keys
            for key in ('accepting_states', 'F', 'acc_sets', 'final_states', 'acc'):
                if key in G.graph:
                    val = G.graph[key]
                    # handle dict-of-lists for 'acc_sets'
                    if isinstance(val, dict):
                        for subset in val.values():
                            accepting.update(subset)
                    # handle list/tuple/set
                    elif isinstance(val, (list, tuple, set)):
                        accepting.update(val)
                    break
            # 3b) node-level flags
            if not accepting:
                for n, data in G.nodes(data=True):
                    if (data.get('accept', False)
                            or data.get('accepting', False)
                            or data.get('final', False)
                            or n.lower().startswith('accept')):
                        accepting.add(n)
            # 3c) warn if still none
            if not accepting:
                print(f"Warning: no accepting states found for automaton '{name}'")

            # 4) store metadata on the graph
            G.graph['initial_state'] = q0
            G.graph['accepting_states'] = accepting

            # 5) attach as attributes for easy access
            G.initial_state = q0
            G.accepting_states = accepting

            # 6) implement the transition function
            def delta(state: Any, inputs: Set[str], G=G) -> Any:
                """
                From `state`, follow any outgoing edge whose label set is a subset of inputs.
                If none match, remain in `state`.
                """
                for _, nxt, data in G.out_edges(state, data=True):
                    lbl = data.get('label', set())
                    lbl_set: Set[str] = set(lbl) if not isinstance(lbl, set) else lbl
                    if lbl_set <= inputs:
                        return nxt
                return state

            G.delta = delta

            # Step 7: Build disjunctive transition groups
            # Result: for each state, list of AP sets that all trigger the same q → q′
            disj_eq: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

            for state in G.nodes:
                # Map from target state to set of APs that lead there
                transitions: Dict[Any, Set[str]] = defaultdict(set)

                for _, tgt, data in G.out_edges(state, data=True):
                    label = data.get("label", set())
                    label_set: Set[str] = {label} if isinstance(label, str) else set(label)
                    for ap in label_set:
                        transitions[tgt].add(ap)

                for tgt_state, aps in transitions.items():
                    if len(aps) > 1:
                        disj_eq[state].append({"next": tgt_state, "aps": aps})

            # Attach to DFA graph metadata
            G.graph["disjunctive_equivalents"] = dict(disj_eq)

            automata[name] = G

    return automata
