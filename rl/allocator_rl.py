# rl/allocator_rl.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional

# 프로젝트 내부 모듈
from ltl_core.agent import Agent
from ltl_core.specification import Specification
from ltl_core.binding_manager import BindingManager
from ltl_core.labeler import Labeler
from ltl_core.workspace import Workspace
from ltl_core.value_fn import ValueBank

# 기존 동작을 1:1로 보장하기 위해, 최종 배포된 단순 할당기를 그대로 사용
# (당신 프로젝트의 위치에 맞춰 import 경로가 'allocator' 입니다)
from allocator import RandomAllocator

# --- 선택형: value 기반 기능을 나중에 다시 켜고 싶을 때 사용 ---
# from rl.value_features import build_s_vector

class RLAllocator:
    """
    SAFE/COMPAT 모드:
      - 기본값 compat=True: 모든 행동 결정을 기존 RandomAllocator에 위임.
      - compat=False 로 주면, 이후 value 기반 로직을 점진적으로 다시 켤 수 있음.
    이 파일은 '행동 결정'만 넘겨주므로, 기존 Nav→Scan→Verify 파이프라인이 즉시 복구됩니다.
    """

    def __init__(
        self,
        spec: Specification,
        agents_by_type: Dict[str, List[Agent]],
        binding_manager: BindingManager,
        labeler: Labeler,
        workspace: Workspace,
        *,
        value_bank: Optional[ValueBank] = None,
        eta_weight: float = 1.0,
        dv_weight: float = 0.0,
        compat: bool = True,   # <-- 기본값: 예전 동작 100% 복구
    ):
        self.spec = spec
        self.agents_by_type = agents_by_type
        self.binding_manager = binding_manager
        self.labeler = labeler
        self.ws = workspace

        self.value_bank = value_bank
        self.eta_weight = float(eta_weight)
        self.dv_weight = float(dv_weight)

        # 기존 단순 할당기 그대로 보유
        self._baseline = RandomAllocator(spec, agents_by_type, binding_manager, labeler, workspace)

        # 호환 모드 플래그
        self.compat = bool(compat)

        # (나중에 compat=False로 켜고 싶은 분들을 위한 자리만 남겨둠)
        # self._use_value = not self.compat

    # Simulation.step(...)에서 부르는 동일한 엔트리포인트
    def choose_eta(
        self, unlocked: Set[str], completed: List[str], aps: Set[str]
    ) -> Dict[Agent, str]:
        """
        compat=True  : 기존 RandomAllocator.choose_eta(...)에 100% 위임
        compat=False : (향후) value 기반 선택을 점진적으로 켜는 자리
        """
        if self.compat or (self.value_bank is None):
            # === 즉시 복구 경로 ===
            return self._baseline.choose_eta(unlocked, completed, aps)

        # === (옵션) value 기반 로직을 다시 붙일 때 여기에 구현 ===
        # 현재는 안전을 위해 호환 모드만 활성화
        return self._baseline.choose_eta(unlocked, completed, aps)
