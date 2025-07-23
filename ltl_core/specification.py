import re
from .symbolic_function import SymbolicFunction
from ltl_core.dag_builder import build_dag
from ltl_core.automaton_generator import compile_automata

ENVIRONMENT_AP_PREFIXES = ["p_found", "p_notfound", "p_verified"]
AP_TYPE_PREFIX_MAP = {
    "p_nav": "physical",
    "p_scan": "symbolic",
    "p_pickup": "physical",
    "p_dropoff": "physical",
    "p_verify": "symbolic",
    "p_monitor": "symbolic",
    "p_rank": "symbolic",
    "p_submit": "symbolic",
}


def get_ap_prefix(ap_name: str) -> str:
    return "_".join(ap_name.split("_")[:2])


def is_environment_ap(ap: str) -> bool:
    """Returns True if the AP starts with any known environment-driven prefix."""
    return any(ap.startswith(prefix + "_") for prefix in ENVIRONMENT_AP_PREFIXES)


class Specification:
    def __init__(self):
        self.case = []
        self.hierarchy = []
        self.binding_manager = None
        self.dag = []
        self.automata = []

    def get_task_specification(self, case, s=None, binding_manager=None):
        self.case = case
        self.binding_manager = binding_manager

        if s is not None:
            self.get_sar_specification_with_mask(case, s, binding_manager=binding_manager)
        else:
            self.get_sar_specification(case)

        self.dag = build_dag(self)
        self.automata = compile_automata(self)

    def get_sar_specification_with_mask(self, case, s, binding_manager=None):
        # Use the passed-in binding manager, or fallback to internal one
        bm = binding_manager or self.binding_manager
        self.binding_manager = bm  # store it for access later

        hierarchy = []
        target_ids = [i for i, active in enumerate(s) if active]

        target_id_map = {}
        current_id = 0

        physical_target_ids = []
        for i in target_ids:
            target_id_map[f"region_{i}"] = current_id
            physical_target_ids.append(current_id)
            current_id += 1

        for label in ["monitor", "rank", "submit"]:
            target_id_map[label] = current_id
            current_id += 1

        if case == "Case2":
            level_one = {
                "p_0": "<> p_101 && <> p_102"
            }
            hierarchy.append(level_one)

            level_two = {
                "p_102": "<> p_oversight"
            }
            # Insert auxiliary p_101_i nodes for each target
            for tid in physical_target_ids:
                level_two[f"p_101_aux_{tid}"] = (
                    f"<> (p_navscan_{tid} && <> (p_confirm_{tid} && <> (p_rescue_{tid} || p_skip_{tid})))"
                )
            # Structural node
            # level_two["p_101"] = " && ".join(f"<> p_101_aux_{tid}" for tid in physical_target_ids)
            # level_two["p_101"] = ""
            level_two["p_101"] = " && ".join(f"p_101_aux_{tid}" for tid in physical_target_ids)

            hierarchy.append(level_two)

            # Level 3: split out nav→scan, then search→verify
            level_three = {}
            for tid in physical_target_ids:
                # enforce nav then scan (drone only)
                level_three[f"p_navscan_{tid}"] = f"<> (p_nav_{tid} && <> p_scan_{tid})"
                # then require that navscan is followed by verify (human only)
                level_three[f"p_confirm_{tid}"] = f"<> p_verify_{tid})"
                # your existing rescue and skip
                level_three[f"p_rescue_{tid}"] = (
                    f"<> (p_verify_{tid} && <> (p_foundgate_{tid} && <> (p_pickup_{tid} && <> p_dropoff_{tid})))")
                level_three[f"p_skip_{tid}"] = f"<> (p_verify_{tid} && <> p_notfoundgate_{tid})"

            # oversight unchanged
            level_three["p_oversight"] = "<> (p_monitor_0 && <> p_submit_0) && <> p_rank_0"
            hierarchy.append(level_three)

            # Level 4: atomic definitions
            level_four = {}
            for tid in physical_target_ids:
                # nav & scan remain drone-only
                level_four[f"p_nav_{tid}"] = f"<> p_nav_{tid}_1_1_0"
                level_four[f"p_scan_{tid}"] = f"<> p_scan_{tid}_1_1_0"
                # verify now human-only
                level_four[f"p_verify_{tid}"] = f"<> p_verify_{tid}_3_1_0"

                level_four[f"p_foundgate_{tid}"] = f"<> p_found_{tid}_0_0_0"
                level_four[f"p_notfoundgate_{tid}"] = f"<> p_notfound_{tid}_0_0_0"
                level_four[f"p_pickup_{tid}"] = f"<> p_pickup_{tid}_2_1_0"
                level_four[f"p_dropoff_{tid}"] = f"<> p_dropoff_{tid}_2_1_0"

            # keep monitor/rank/submit as before
            level_four["p_monitor_0"] = (
                f"<> (p_monitor_{target_id_map['monitor']}_1_1_0 "
                f"|| p_monitor_{target_id_map['monitor']}_3_1_0)"
            )
            level_four["p_rank_0"] = (
                f"<> (p_rank_{target_id_map['rank']}_1_1_0 "
                f"|| p_rank_{target_id_map['rank']}_3_1_0)"
            )
            level_four["p_submit_0"] = (
                f"<> (p_submit_{target_id_map['submit']}_1_1_0 "
                f"|| p_submit_{target_id_map['submit']}_3_1_0)"
            )

            hierarchy.append(level_four)
            self.hierarchy = hierarchy

            # ------------------ BINDING REGISTRATION ------------------
            for parent, formula in level_three.items():
                symbolic_children = re.findall(r"p_[a-zA-Z]+_[0-9A-Z]+", formula)
                ap_subexprs = [level_four[sc] for sc in symbolic_children if sc in level_four]
                aps = set(re.findall(r"p_[a-zA-Z]+_[0-9A-Z]+_[0-9]+_[0-9]+_[0-9]+", " ".join(ap_subexprs)))

                per_type_tasks = {}
                for ap in aps:
                    parts = ap.split("_")
                    if len(parts) >= 6:
                        agent_type = int(parts[3])
                        role = self.resolve_role(agent_type)
                        per_type_tasks.setdefault(role, []).append(ap)

                group_tasks = []
                group_roles = set()
                for role, tasks in per_type_tasks.items():
                    for ap in tasks:
                        if not is_environment_ap(ap):
                            group_tasks.append(ap)
                            group_roles.add(role)

                if group_tasks:
                    bm.register_group(group_key=parent, tasks=group_tasks, group_type=group_roles)

    def get_sar_specification(self, case):
        hierarchy = []
        if case == 0:
            level_one = {"p0": "<> (p100_2_1_0 || p200_1_1_0) && <> p300_1_1_0"}
            hierarchy.append(level_one)

            level_two = {
                "p100": "<> (p2_2_1_0 && <> p3_1_1_0)",
                "p200": "<> p4_1_1_0",
                "p300": "<> p5_1_1_0"
            }
            hierarchy.append(level_two)
        elif case == 1:
            level_one = {"p0": "<> (p100_1_1_0 && <> p1_2_1_0)"}
            hierarchy.append(level_one)

            level_two = {
                "p100": "<> (p2_1_1_0 && <> p4_1_1_0)"
            }
            hierarchy.append(level_two)
        self.hierarchy = hierarchy

    def print_sar_specification(self):
        for level, formulas in enumerate(self.hierarchy, start=1):
            print(f"\n--- Level {level} ---")
            for name, ltl in formulas.items():
                print(f"{name}: {ltl}")

    def extract_symbolic_functions(self):
        pattern = r"p_([a-zA-Z]+)_([0-9A-Z]+)_([0-9]+)_([0-9]+)"
        unique_aps = {}

        for level_index, level in enumerate(self.hierarchy):
            current_level = level_index + 1
            for _, formula in level.items():
                matches = re.findall(pattern, formula)
                for task_type, target, agent_type, instance in matches:
                    ap_id = f"p_{task_type}_{target}_{agent_type}_1_{instance}"
                    if ap_id not in unique_aps:
                        sf = SymbolicFunction(
                            task_label=f"{task_type}_{target}",
                            agent_label=None,
                            ap_type=task_type,
                            target=target,
                            role=self.resolve_role(int(agent_type)),
                            level=current_level,
                            ap_id=ap_id
                        )
                        unique_aps[ap_id] = sf

        return list(unique_aps.values())

    @staticmethod
    def resolve_role(agent_type: int):
        return {
            1: "drone",
            2: "gv",
            3: "human"
        }.get(agent_type, "unknown")

    def get_all_function_names(self):
        atomic_pattern = r"p_([a-zA-Z]+)_([0-9A-Z]+)_([0-9]+)_([0-9]+)_([0-9]+)"
        general_pattern = r"p_[a-zA-Z0-9_]+"

        composite_names = set()
        atomic_names = set()

        for level in self.hierarchy:
            for key, formula in level.items():
                composite_names.add(key)
                found = re.findall(general_pattern, formula)
                for name in found:
                    if re.match(atomic_pattern, name):
                        atomic_names.add(name)
                    else:
                        composite_names.add(name)

        composite_names -= atomic_names
        return sorted(composite_names), sorted(atomic_names)

    def get_required_role_by_ap(self, ap_name: str) -> str:
        pattern = r"p_[a-zA-Z]+_[0-9A-Z]+_([0-9])_[0-9]+_[0-9]+"
        match = re.match(pattern, ap_name)
        if match:
            agent_type = int(match.group(1))
            return self.resolve_role(agent_type)
        else:
            return "unknown"
