"""전환 시점의 각 기억 프레임을 같은 회차 Native 기억과 비교하는 R².

maskmem_features와 obj_ptr를 각각 펼쳐 R² = 1 - SSE / SST를 구한다.
회복률의 Native 반복 중앙값과 별개로, 복원율은 같은 회차의 실제 Native tensor를 쓴다.
대표값은 준비된 Target 기억 전체의 필드별 R²이다. 칸별 R² 단순 평균과 구분한다.
칸별 원점수는 보존하며 영상/회차 집계는 tables/summaries.py에서 수행한다.
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


def bank_score(points, field):
    """준비된 Target의 전체 기억 원소를 이어 붙인 R²을 충분통계량으로 재계산한다.

    Native에만 있는 칸은 Target에 전달되지 않았으므로 대표값의 대상이 아니다.
    Target 칸 하나라도 대응 Native/필드가 없으면 전체 점수는 N/A이다.
    """
    selected = [p[field] for p in points if p.get('target_is_cond') is not None]
    count, center, sst, sse, compared = 0, 0., 0., 0., 0
    for part in selected:
        n = part.get('n_elements', 0)
        if not n or any(part.get(k) is None for k in ('sse', 'sst', 'native_mean')):
            continue
        delta = part['native_mean'] - center
        total = count + n
        # 칸별 Native 평균이 다르면 평균 사이의 변동도 전체 SST에 포함한다.
        sst += part['sst'] + delta ** 2 * count * n / total
        center += delta * n / total
        count = total
        sse += part['sse']
        compared += 1
    complete = bool(selected) and compared == len(selected)
    finite = all(math.isfinite(v) for v in (center, sst, sse))
    r2 = 1 - sse / sst if complete and finite and count >= 2 and sst > 0 else None
    if r2 is not None and not math.isfinite(r2):
        r2 = None
        finite = False
    status = ('empty_target_memory' if not selected else 'incomplete_memory' if not complete
              else 'nonfinite_statistics' if not finite else 'insufficient_elements' if count < 2
              else 'zero_native_variance' if sst == 0 else 'ok')
    return {'r2': r2, 'status': status, 'sse': sse if finite else None,
            'sst': sst if finite else None, 'native_mean': center if count and finite else None,
            'n_elements': count, 'target_frame_count': len(selected), 'compared_frame_count': compared,
            'missing_or_invalid_frame_count': len(selected) - compared,
            'native_only_frame_count': len(points) - len(selected)}


def recompute(rows):
    """저장된 충분통계량으로 대표값을 만든다. 원본 줄/기억 프레임 값은 바꾸지 않는다."""
    out = []
    for row in rows:
        result = dict(row)
        result['restoration_statistic'] = 'full_prepared_memory'
        result['restoration_summary'] = {}
        for field in FIELDS:
            summary = bank_score(row.get('restoration_frame_scores', []), field)
            if row.get('restoration_status') == 'source_only_no_target_memory':
                summary['status'] = 'source_only_no_target_memory'
            result[f'r2_{field}'] = summary['r2']
            result['restoration_summary'][field] = summary
        out.append(result)
    return out
