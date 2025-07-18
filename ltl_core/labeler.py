import re
import random
from typing import Any, Dict, List, Set
from .specification import get_ap_prefix, AP_TYPE_PREFIX_MAP, is_environment_ap
import networkx as nx


class Labeler:
    """
    Tracks decomposed-LTL formulas φ(k,i) via DFA states, extracts atomic
    propositions (APs) from the environment, and classifies formulas
    (levels 1–4) as Completed, Unlocked, or Locked.

    Leaf (level-4) formulas correspond one-to-one to APs and are marked
    Completed immediately when their AP is satisfied. Composite formulas
    (levels 1–3) are updated bottom-up once all children are complete.
    """

    def __init__(self, spec, dag: nx.DiGraph, automata: Dict[str, Any]):
        self.spec = spec
        self.dag = dag
        self.automata = automata

        # All formula names (levels 1–4)
        self.formulas = list(automata.keys())
        # Level-4 formulas
        self.leafs = list(spec.hierarchy[-1].keys())
        # Map each leaf formula to the set of AP tokens it uses
        self.leaf_APs_map: Dict[str, Set[str]] = {
            leaf: set(re.findall(r"p_[A-Za-z0-9_]+", spec.hierarchy[-1][leaf]))
            for leaf in self.leafs
        }

        # Predecessor maps from build_dag
        self.preds_map = dag.graph["predecessors_map"]
        self.pair_preds = dag.graph["pairwise_predecessors"]

        # Top-level formula name (assumed to be the only Level 1 entry)
        self.top_level_formula = list(spec.hierarchy[0].keys())[0]

        # Initialise DFA states
        self.states: Dict[str, Any] = {}
        for name, dfa in automata.items():
            if hasattr(dfa, "initial_state"):
                q0 = dfa.initial_state
            elif "initial_state" in getattr(dfa, "graph", {}):
                q0 = dfa.graph["initial_state"]
            else:
                indeg = dict(dfa.in_degree())
                q0 = next((n for n, d in indeg.items() if d == 0), None)
            if q0 is None:
                raise RuntimeError(f"DFA '{name}' missing initial_state")
            self.states[name] = q0

        self.current_aps: Set[str] = set()
        self._completed: Set[str] = set()
        self.unlocked: List[str] = []
        self.locked: List[str] = []

        # Propagate DFA for each group key (if it has a corresponding DFA)
        for group_key in self.spec.hierarchy[2].keys():
            if group_key in self.automata:
                if hasattr(self.spec, "binding_manager") and self.spec.binding_manager:
                    self.spec.binding_manager.group_to_automaton[group_key] = self.automata[group_key]

        self.binding_manager = spec.binding_manager

        self.chosen_gate_per_group = {}  # key = group, value = chosen gate

    # ------------------------------------------------------------------ #
    # Public helpers
    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset DFA states and clear AP / completion tracking."""
        for name, dfa in self.automata.items():
            if hasattr(dfa, "initial_state"):
                self.states[name] = dfa.initial_state
            elif "initial_state" in getattr(dfa, "graph", {}):
                self.states[name] = dfa.graph["initial_state"]
            else:
                indeg = dict(dfa.in_degree())
                self.states[name] = next((n for n, d in indeg.items() if d == 0), None)

        self.current_aps = set()
        self._completed = set()
        self._update_unlocked()
        self._update_locked()

    def extract_APs(self, state) -> Set[str]:
        """
        Convert the workspace state into a set of currently true atomic propositions (APs),
        including physical arrival checks and symbolic progress.
        """
        base_aps: Set[str] = state.get_true_aps()
        valid_aps: Set[str] = set()

        # 1. Include environment-driven APs directly
        env_aps = {ap for ap in base_aps if is_environment_ap(ap)}
        valid_aps.update(env_aps)

        for ap in base_aps:
            prefix = get_ap_prefix(ap)
            ap_type = AP_TYPE_PREFIX_MAP.get(prefix)

            if ap_type == "physical":
                target_idx = self._extract_target_index(ap)
                goal = state.target_locations[target_idx]

                group = self.binding_manager.task_to_group.get(ap)
                agent_type = self._get_agent_type(ap)
                agent = self.binding_manager.get_bound_agent_for_group(group, agent_type=agent_type)

                if agent is None:
                    continue

                if agent.has_arrived():
                    valid_aps.add(ap)

            elif ap_type == "symbolic":
                group = self.binding_manager.task_to_group.get(ap)
                agent_type = self._get_agent_type(ap)
                agent = self.binding_manager.get_bound_agent_for_group(group, agent_type=agent_type)

                if agent and agent.has_completed(ap):
                    valid_aps.add(ap)

            else:
                print(f"[WARN] Unrecognized AP type: {ap}")

        self.current_aps = valid_aps
        return valid_aps

    @staticmethod
    def _extract_target_index(ap: str) -> int:
        # e.g., p_nav_0_1_1_0 -> 0
        return int(ap.split("_")[2])

    @staticmethod
    def _get_agent_type(ap: str) -> str:
        # e.g., p_nave_0_1_1_0 -> "drone" "if type = 1"
        role_map = {"1": "drone", "2": "gv", "3": "human"}
        return role_map.get(ap.split("_")[3], "unknown")

    def empty_extract_APs(self, state) -> Set[str]:
        """
        Convert raw *state* from the environment into the set of AP names
        that are true at this timestep.
        """
        aps: Set[str] = set()
        # Example: if state.drone_pos == spec.locations['nav_1']: aps.add('p_nav_1_1_1_0')
        self.current_aps = aps
        return aps

    def advance(self, true_APs: Set[str]) -> None:
        """Advance the labeler one step, given the set of true AP tokens."""
        updated: Set[str] = set()

        # Step 1 — advance DFA transitions for atomic nodes that have a DFA
        for node in self.dag.nodes:
            if node in self.automata and self._is_atomic_node(node):
                state = self.states[node]
                next_state = state
                advanced = False

                for _, tgt, data in self.automata[node].out_edges(state, data=True):
                    label = data.get("label")
                    label_set: Set[str] = {label} if isinstance(label, str) else set(label)

                    if label_set & true_APs:
                        next_state = tgt
                        advanced = True
                        break

                self.states[node] = next_state

                if self._is_accepting(self.automata[node], next_state):
                    if node not in self._completed:
                        self._completed.add(node)
                        updated.add(node)

        # Step 2 — environment-driven atomic nodes with no DFA (p_found, p_notfound, etc.)
        for node in self.dag.nodes:
            if self._is_atomic_node(node) and node not in self.automata:
                if node in true_APs and node not in self._completed:
                    self._completed.add(node)
                    updated.add(node)

        # Step 3 — env-disjunctive gates (e.g., p_foundgate_0_0_0_0 or p_notfoundgate_0_0_0_0)
        gates_by_group = {}
        for node in self.dag.nodes:
            if node.startswith(("p_foundgate_", "p_notfoundgate_")):
                if node in self.unlocked and node not in self._completed:
                    group = "_".join(node.split("_")[2:6])  # e.g., 0_0_0_0
                    gates_by_group.setdefault(group, []).append(node)

        # Found or Not found
        for group, gate_list in gates_by_group.items():
            if group in self.chosen_gate_per_group:
                chosen = self.chosen_gate_per_group[group]
            else:
                # Random
                chosen = random.choice(gate_list)
                # Fixed: found
                # chosen = next((g for g in gate_list if g.startswith("p_foundgate_")), gate_list[0])
                self.chosen_gate_per_group[group] = chosen
            if chosen not in self._completed:
                self._completed.add(chosen)
                updated.add(chosen)

        # Step 4 — propagate completions upward in the DAG
        self._propagate_completions(updated)

        # Step 5 — refresh unlocked / locked lists
        self._update_unlocked()
        self._update_locked()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _propagate_completions(self, changed_nodes: Set[str]):
        """
        Bottom-up propagation that respects AND / OR semantics.

        A composite node becomes *completed* when:
        - every child in ``mandatory`` is completed, and
        - for each alt-group in ``alt_groups`` at least one child is completed.
        """
        stack, visited = list(changed_nodes), set()

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            for parent in self.dag.predecessors(node):
                if parent in self._completed:        # already done
                    continue
                if self._is_parent_completed(parent):
                    self._completed.add(parent)
                    stack.append(parent)

    def _is_atomic_node(self, node: str) -> bool:
        """
        A node is atomic if it is (a) one of the concrete AP tokens in the DAG
        or (b) a level-4 formula name.
        """
        atomic_tokens = set(self.dag.graph.get("atomic_names", []))
        leaf_formulas = set(self.leafs)
        return node in atomic_tokens or node in leaf_formulas

    def _update_unlocked(self):
        """
        Re-compute the list of unlocked formulas. We iterate at most |V|
        times to avoid infinite propagation loops.
        """
        self.unlocked = []
        completed = set(self._completed)
        unlocked_set: Set[str] = set()

        for _ in range(len(self.dag.nodes)):
            changed = False
            for node in self.dag.nodes:
                if node in completed or node in unlocked_set:
                    continue

                temporal_preds = set(self.pair_preds.get(node, []))
                # All temporal predecessors must be completed
                if temporal_preds and not temporal_preds <= completed:
                    continue

                # Structural parents may be either unlocked or completed
                structural_parents = set(self.preds_map.get(node, [])) - temporal_preds
                parents_ok = all(
                    p in unlocked_set or p in completed for p in structural_parents
                )
                if not parents_ok:
                    continue

                unlocked_set.add(node)
                changed = True

            if not changed:
                break

        self.unlocked = list(unlocked_set)

    def _update_locked(self):
        """Formulas that are neither completed nor unlocked."""
        completed = self._completed
        unlocked = set(self.unlocked)
        self.locked = [n for n in self.formulas if n not in completed and n not in unlocked]

    def get_group_ordered_tasks(self, group):
        """Return ordered list of APs in a group using pairwise order constraints"""
        if not self.binding_manager:
            return []

        tasks = list(self.binding_manager.group_to_tasks.get(group, []))
        if not tasks:
            return []

        predecessors = {task: [] for task in tasks}
        for task in tasks:
            preds = self.pair_preds.get(task, [])
            for pred in preds:
                if pred in tasks:
                    predecessors[task].append(pred)

        ordered = []
        visited = set()

        def visit(t):
            if t in visited:
                return
            for p in predecessors[t]:
                visit(p)
            ordered.append(t)
            visited.add(t)

        for t in tasks:
            visit(t)

        return ordered

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #

    def get_completed(self) -> List[str]:
        return list(self._completed)

    def get_unlocked(self) -> List[str]:
        return self.unlocked

    def get_locked(self) -> List[str]:
        return self.locked

    def get_unlocked_leaf_formulas(self) -> List[str]:
        leaf_set = set(self.leafs)
        return [f for f in self.unlocked if f in leaf_set]

    def get_unlocked_APs(self) -> Set[str]:
        aps: Set[str] = set()

        # 1) team-driven leaves
        for leaf in self.get_unlocked_leaf_formulas():
            if leaf.startswith(("p_foundgate_", "p_notfoundgate_")):
                continue  # handled below
            ap_candidates = self.leaf_APs_map.get(leaf, set())
            for ap in ap_candidates:
                preds = self.pair_preds.get(ap, [])
                if all(p in self._completed for p in preds):
                    aps.add(ap)

        # 2) env-driven gates (p_foundgate_* and p_notfoundgate_*) — disjunctive logic
        gates_by_group = {}

        for leaf in self.leafs:
            if leaf.startswith(("p_foundgate_", "p_notfoundgate_")):
                if leaf in self.unlocked and leaf not in self._completed:
                    group = "_".join(leaf.split("_")[2:6])  # e.g., 0_0_0_0
                    gates_by_group.setdefault(group, []).append(leaf)

        for group, gate_list in gates_by_group.items():
            chosen_gate = sorted(gate_list)[0]  # deterministic: pick p_foundgate over p_notfoundgate if both unlocked
            aps.update(self.leaf_APs_map.get(chosen_gate, set()))

        return aps

    # ------------------------------------------------------------------ #
    # DFA helpers
    # ------------------------------------------------------------------ #

    def get_disjunctive_ap_groups(self) -> List[Set[str]]:
        """
        Return all sets of disjunctive APs (regardless of DFA state).
        These are APs that appear together in a disjunctive transition group.
        """
        disj_groups: List[Set[str]] = []
        seen_groups: Set[frozenset] = set()

        for dfa in self.automata.values():
            equiv = dfa.graph.get("disjunctive_equivalents", {})
            for entries in equiv.values():  # entries is a list of {aps, next}
                for entry in entries:
                    aps = entry.get("aps", set())
                    if len(aps) > 1:
                        fs = frozenset(aps)
                        if fs not in seen_groups:
                            disj_groups.append(set(aps))
                            seen_groups.add(fs)

        return disj_groups

    def _step_dfa(self, dfa: Any, state: Any, inputs: Set[str]) -> Any:
        """
        Step a DFA manually if needed; returns next state.
        """
        if hasattr(dfa, "delta"):
            return dfa.delta(state, inputs)

        for _, nxt, data in dfa.out_edges(state, data=True):
            lbl = data.get("label", set())
            lbl_set = {lbl} if isinstance(lbl, str) else set(lbl)
            if lbl_set <= inputs:
                return nxt
        return state

    @staticmethod
    def _is_accepting(dfa: Any, state: Any) -> bool:
        """
        Test whether a DFA state is accepting.
        """
        if hasattr(dfa, "accepting_states"):
            return state in dfa.accepting_states
        if isinstance(getattr(dfa, "graph", None), dict):
            if "accepting_states" in dfa.graph:
                return state in dfa.graph["accepting_states"]
        return dfa.nodes[state].get("accepting", False)

    # ------------------------------------------------------------------ #
    # Completion rules with AND/OR children
    # ------------------------------------------------------------------ #
    def _is_parent_completed(self, parent: str) -> bool:
        """
        Return True ⇔ *all* mandatory children are complete **and**
        for every alternative-group, at least one member is complete.
        """
        mandatory = self.dag.nodes[parent].get("mandatory", [])
        alt_groups = self.dag.nodes[parent].get("alt_groups", [])

        # mandatory children
        if any(c not in self._completed for c in mandatory):
            return False

        # each OR-group must contribute ≥ 1 completed child
        for group in alt_groups:
            if not any(c in self._completed for c in group):
                return False

        return True

    def all_completed(self) -> bool:
        """
        Return True if the top-level formula (typically p_0) is completed.
        This avoids being stuck on optional branches.
        """
        return self.top_level_formula in self._completed
