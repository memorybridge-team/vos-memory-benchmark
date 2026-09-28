"""[추가] 지표. 확정 표에는 넣지 않고 추가 표에만 나온다. 결과 열 이름은 모두 extra_ 로 시작.

전환 뒤 프레임만 본다 (정답이 모든 프레임에 있는 데이터셋만).

extra_jf_at_n             J&F@N: 전환 뒤 객체가 보이는 처음 N 프레임의 J&F — 넘긴 직후 충격
extra_switch_shock        switch shock: 위 값 − 전환 직전 보이는 N 프레임(Small)의 J&F. 음수 = 떨어짐
extra_id_switch_rate      ID 뒤바뀜: 보이는 프레임 중, 예측이 자기보다 다른 객체와 더 겹친 비율
extra_recovery_frames     회복 지연: 전환 뒤 처음 보이는 프레임부터 J ≥ 기준이 될 때까지 걸린 프레임
extra_absent_false_alarm  부재 오검출: 정답에 객체가 없는 프레임 중 무언가를 칠한 비율
extra_failure             실패: 전환 뒤 보이는 프레임의 평균 J 가 기준보다 낮으면 1
"""

from __future__ import annotations

import numpy as np

import settings

KEYS = ("extra_jf_at_n", "extra_switch_shock", "extra_id_switch_rate",
        "extra_recovery_frames", "extra_absent_false_alarm", "extra_failure")


def other_object_iou(pred, labels, ignore, obj_id) -> float:
    """예측이 다른 객체 정답과 가장 많이 겹친 IoU."""
    if ignore is not None:
        pred = pred & ~ignore
    if not pred.any():
        return 0.0
    best = 0.0
    for other in np.unique(labels[pred]):
        if other == 0 or other == obj_id:
            continue
        gt = labels == other
        best = max(best, (pred & gt).sum() / (pred | gt).sum())
    return float(best)


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def compute(pre: dict, post: dict) -> dict:
    """pre = {프레임: FrameScore} 전환 전 (Small), post = 전환 뒤 (방법)."""
    n = settings.EXTRA_JF_AT
    visible = [f for f in sorted(post) if post[f].gt_visible]
    absent = [f for f in sorted(post) if not post[f].gt_visible]
    pre_visible = [f for f in sorted(pre) if pre[f].gt_visible]

    jf_at_n = _mean([post[f].jf for f in visible[:n]])
    pre_jf = _mean([pre[f].jf for f in pre_visible[-n:]])
    shock = jf_at_n - pre_jf if jf_at_n is not None and pre_jf is not None else None

    id_switch = _mean([post[f].other_iou >= settings.EXTRA_ID_SWITCH_IOU
                       and post[f].other_iou > post[f].j for f in visible])

    recovery = None
    for f in visible:
        if post[f].j >= settings.EXTRA_RECOVERY_J:
            recovery = f - visible[0]
            break

    mean_j = _mean([post[f].j for f in visible])
    failure = None if mean_j is None else float(mean_j < settings.EXTRA_FAILURE_J)

    return {
        "extra_jf_at_n": jf_at_n,
        "extra_switch_shock": shock,
        "extra_id_switch_rate": id_switch,
        "extra_recovery_frames": recovery,
        "extra_absent_false_alarm": _mean([post[f].pred_any for f in absent]),
        "extra_failure": failure,
    }
