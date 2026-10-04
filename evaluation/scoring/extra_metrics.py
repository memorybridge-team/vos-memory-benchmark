"""[추가] 지표. 확정 표에는 넣지 않고 추가 표에만 나온다. 결과 열 이름은 모두 extra_ 로 시작.

실패 분석 (전환 뒤, 정답에 객체가 보이는 프레임만)
  extra_failure_rate    실패 비율 = 1 − J_Recall
                        J_Recall = (J > EXTRA_RECALL_J 인 프레임 수) ÷ (그 프레임 수)

결과 분석
  extra_agreement       출력 일치도: 전환 뒤 모든 프레임에서 방법 마스크와 Full Replay 마스크의 IoU 평균

비용 세부 extra_switch_gpu_mb 는 evaluation/cost.py.
"""

from __future__ import annotations

import numpy as np

import settings
from evaluation.scoring.main_metrics import mean


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """겹친 넓이 ÷ 합친 넓이. 둘 다 비어 있으면 1."""
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def failure_rate(scores: list):
    """scores = 전환 뒤 보이는 프레임의 FrameScore → 실패 비율 (프레임이 없으면 None)."""
    recall = mean(s.j > settings.EXTRA_RECALL_J for s in scores)
    return None if recall is None else 1 - recall


def agreement(ious: dict, switch_frame: int):
    """ious = {프레임: Full Replay 마스크와의 IoU} → 전환 뒤 평균."""
    return mean(iou for frame, iou in ious.items() if frame > switch_frame)
