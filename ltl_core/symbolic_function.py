import re


class SymbolicFunction:
    def __init__(self, task_label, agent_label, ap_type, target=None, role=None, level=None, ap_id=None):
        """
        Represents a symbolic atomic proposition (AP) that evolves over time through agent activity.

        Args:
            task_label (str): Human-readable symbolic task label (e.g., 'scan_0', 'verify_1').
            agent_label (str): Identifier for the assigned agent (e.g., 'D0', 'H1').
            ap_type (str): Type of symbolic task (e.g., 'scan', 'verify', 'submit').
            target (int|str, optional): Associated target region or task ID.
            role (str, optional): Role of the agent ('drone', 'human', 'gv', etc.).
            level (int, optional): Hierarchical level in the LTL decomposition (e.g., level 4).
            ap_id (str, optional): The formal LTL atomic proposition (e.g., 'p_scan_0_1_1_0').
        """
        self.task_label = task_label          # For visualization/debugging
        self.agent_label = agent_label
        self.ap_type = ap_type
        self.target = target
        self.role = role
        self.level = level

        # Formal LTL AP used for DFA/NBA transitions
        self.ap_id = ap_id or self._generate_ltl_ap()

        self.progress = 0.0  # Real-valued progress ∈ [0, 1]
        self.status = "inactive"  # 'inactive' | 'active' | 'complete'

    def _generate_ltl_ap(self):
        """Generates a formal LTL AP name based on task metadata."""
        suffix = f"{self.target}_{self._role_index()}_1_0" if self.target is not None else "x"
        return f"p_{self.ap_type}_{suffix}"

    def _role_index(self):
        return {"drone": 1, "gv": 2, "human": 3}.get(self.role, 0)

    def activate(self):
        """Set the function as ready to begin progressing."""
        if self.status == "inactive":
            self.status = "active"

    def update(self, delta):
        """Only progresses if the function is active."""
        if self.status != "active":
            return  # Do nothing if not yet unlocked
        self.progress = min(1.0, self.progress + delta)
        if self.progress >= 1.0:
            self.status = "complete"

    def reset(self):
        self.progress = 0.0
        self.status = "inactive"

    def is_active(self):
        return self.status == "active"

    def is_complete(self):
        return self.status == "complete"

    def __repr__(self):
        return f"<{self.task_label} ({self.ap_type}) {self.status}: {self.progress:.2f}>"


def extract_symbolic_functions(specification):
    """
    Extracts all atomic propositions (APs) from the full hierarchy (Level 1–4).
    Converts each unique AP of the form p_<label>_<target>_<agent_type>_<req>_<agent_id>
    into a SymbolicFunction.

    Returns:
        List of SymbolicFunction instances.
    """
    ap_set = set()
    symbolic_functions = []

    for level in specification.hierarchy:
        for formula in level.values():
            # Matches e.g. p_nav_0_1_1_0
            matches = re.findall(r"p_\w+_\d+_\d+_\d+_\d+", formula)
            for ap in matches:
                if ap in ap_set:
                    continue
                ap_set.add(ap)

                parts = ap.split("_")  # ['p', 'nav', '0', '1', '1', '0']
                if len(parts) < 6:
                    continue

                ap_type = parts[1]
                target = int(parts[2])
                agent_type = int(parts[3])
                required = int(parts[4])
                agent_id = int(parts[5])
                agent_label = f"A{agent_id}"

                symbolic_functions.append(SymbolicFunction(
                    task_label=f"{ap_type}_{target}",
                    agent_label=agent_label,
                    ap_type=ap_type,
                    target=target,
                    role=None,
                    level=4,
                    ap_id=ap
                ))

    return symbolic_functions

