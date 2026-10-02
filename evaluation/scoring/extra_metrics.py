"""[추가] 지표. 확정 표에는 넣지 않고 추가 표에만 나온다. 결과 열 이름은 모두 extra_ 로 시작.

실패 분석 (전환 뒤, 정답에 객체가 보이는 프레임만 — 정답이 모든 프레임에 있는 데이터셋만)
  extra_id_switch_rate  ID 뒤바뀜: 예측이 자기 물체보다 다른 물체와 더 많이 겹친 프레임 비율 (겹침 = IoU)
  extra_failure         실패: 평균 J 가 EXTRA_FAILURE_J 보다 낮으면 1

언제 잘 되나
  extra_drift_jf        drift: 전환 뒤 경과 프레임 구간(EXTRA_DRIFT_BINS)마다 J&F 평균 (구간 수만큼의 목록)
                        표에서 같은 전환의 Full Replay 값을 빼서 곡선으로 그린다.

결과 분석 (정답이 필요 없어 MOSEv2 valid 에서도 계산)
  extra_agreement       출력 일치도: 전환 뒤 모든 프레임에서 방법 마스크와 Full Replay 마스크의 IoU 평균
"""

from __future__ import annotations

import numpy as np

import settings
from evaluation.scoring.extra_groups import bin_index
from evaluation.scoring.main_metrics import mean


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """겹친 넓이 ÷ 합친 넓이. 둘 다 비어 있으면 1."""
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def other_object_iou(pred, labels, ignore, obj_id) -> float:
    """예측이 다른 객체 정답과 가장 많이 겹친 IoU."""
    if ignore is not None:
        pred = pred & ~ignore
    if not pred.any():
        return 0.0
    others = [o for o in np.unique(labels[pred]) if o not in (0, obj_id)]
    return max((mask_iou(pred, labels == o) for o in others), default=0.0)


def compute(visible: dict, switch_frame: int) -> dict:
    """visible = {프레임: FrameScore} 전환 뒤 정답에 객체가 보이는 프레임만."""
    scores = list(visible.values())
    mean_j = mean(s.j for s in scores)

    drift = [[] for _ in settings.EXTRA_DRIFT_BINS]
    for frame, s in visible.items():
        drift[bin_index(settings.EXTRA_DRIFT_BINS, frame - switch_frame)].append(s.jf)

    return {
        "extra_id_switch_rate": mean(s.other_iou > s.j for s in scores),
        "extra_failure": None if mean_j is None else float(mean_j < settings.EXTRA_FAILURE_J),
        "extra_drift_jf": [mean(values) for values in drift],
    }


def agreement(ious: dict, switch_frame: int):
    """ious = {프레임: Full Replay 마스크와의 IoU} → 전환 뒤 평균."""
    return mean(iou for frame, iou in ious.items() if frame > switch_frame)
