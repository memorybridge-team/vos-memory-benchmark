"""Small → Base+ 로 넘기는 기억 상자.

전환 순간 Small 쪽에서 챙겨 두는 것 전부를 한 상자(HandoffPackage)에 담는다.
비교군마다 이 상자에서 필요한 것만 꺼내 Base+ 에 넘긴다.

    small_memory    SAM2 내부 기억 칸 (Direct State Copy 가 씀)
    prompt_*        처음 받은 정답 마스크 (Original-Prompt, Replay 가 씀)
    last_visible    Small의 마지막 비어 있지 않은 예측 프레임과 마스크 (Original + Last-Visible)

정답은 처음 프롬프트 한 번만 들어간다. 나머지는 모두 Small이 스스로 낸 것.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HandoffPackage:
    switch_frame: int                         # Small이 여기까지 봤다 (Base+ 는 다음 프레임부터)
    prompt_frame: int
    prompt_mask: np.ndarray
    small_memory: dict                        # {프레임: 기억 칸} — sam2_runner.export_memory()
    last_visible: tuple[int, np.ndarray] | None

    def last_visible_or_prompt(self) -> tuple[int, np.ndarray]:
        """Small이 비어 있지 않은 마스크를 한 번도 내지 않았으면 처음 프롬프트를 쓴다."""
        if self.last_visible is not None:
            return self.last_visible
        return self.prompt_frame, self.prompt_mask
