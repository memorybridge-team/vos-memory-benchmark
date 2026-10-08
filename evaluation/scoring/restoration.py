"""전환 시점의 각 기억 프레임을 같은 회차 Native 기억과 비교하는 R².

maskmem_features와 obj_ptr를 각각 펼쳐 R² = 1 - SSE / SST를 구한다.
회복률의 Native 반복 중앙값과 별개로, 복원율은 같은 회차의 실제 Native tensor를 쓴다.
TODO: 칸/객체/영상/회차 평균과 논문 표·그래프의 표현 방식은 미정. 현재 집계하지 않는다.
"""

import math

import torch

FIELDS = ('maskmem_features', 'obj_ptr')
REVISION = 1


def tensor_score(native, prepared):
    """한 필드의 tensor 원소를 펼친 R²과 재집계용 충분통계량. 계산은 CPU float64."""
    result = {'r2': None, 'sse': None, 'sst': None, 'native_mean': None,
              'n_elements': 0,
              'native_shape': list(native.shape) if native is not None else None,
              'target_shape': list(prepared.shape) if prepared is not None else None}
    if native is None or prepared is None:
        return {**result, 'status': 'missing_native_tensor' if native is None else 'missing_target_tensor'}
    if tuple(native.shape) != tuple(prepared.shape):
        return {**result, 'status': 'shape_mismatch'}
    y = native.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
    pred = prepared.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
    if not y.numel():
        return {**result, 'status': 'empty_tensor'}
    if not bool(torch.isfinite(y).all() and torch.isfinite(pred).all()):
        return {**result, 'status': 'nonfinite_tensor'}
    center = float(y.mean())
    sst = float(((y - center) ** 2).sum())
    sse = float(((pred - y) ** 2).sum())
    if not all(math.isfinite(v) for v in (center, sst, sse)):
        return {**result, 'status': 'nonfinite_statistics'}
    result.update(sse=sse, sst=sst, native_mean=center, n_elements=y.numel())
    if y.numel() < 2:
        return {**result, 'status': 'insufficient_elements'}
    if sst == 0:
        return {**result, 'status': 'zero_native_variance'}
    r2 = 1 - sse / sst
    if not math.isfinite(r2):
        return {**result, 'status': 'nonfinite_r2'}
    return {**result, 'r2': r2, 'status': 'ok'}


def frame_scores(prepared, native, switch_frame):
    """전환 때 두 memory bank에 남아 있는 기억 프레임을 번호로 대응한다.

    누락 칸은 N/A와 사유로 남긴다. 객체 가시성/마스크 성공 여부로 기억을 삭제하지 않는다.
    """
    points = []
    for frame in sorted(set(native) | set(prepared)):
        reference, actual = native.get(frame), prepared.get(frame)
        point = {'frame': frame, 'frames_before_switch': switch_frame - frame,
                 'native_is_cond': reference.get('is_cond') if reference is not None else None,
                 'target_is_cond': actual.get('is_cond') if actual is not None else None}
        for field in FIELDS:
            score = tensor_score(reference.get(field) if reference is not None else None,
                                 actual.get(field) if actual is not None else None)
            if reference is None:
                score['status'] = 'missing_native_frame'
            elif actual is None:
                score['status'] = 'missing_target_frame'
            point[field] = score
        points.append(point)
    return points


def result(prepared, native, switch_frame):
    return {'restoration_revision': REVISION, 'restoration_reference': 'native_same_run',
            'restoration_measured_at': switch_frame, 'restoration_status': 'measured' if native or prepared else 'empty_memory',
            'restoration_frame_scores': frame_scores(prepared, native, switch_frame)}


def not_applicable(switch_frame):
    return {'restoration_revision': REVISION, 'restoration_reference': 'native_same_run',
            'restoration_measured_at': switch_frame, 'restoration_status': 'source_only_no_target_memory',
            'restoration_frame_scores': []}
