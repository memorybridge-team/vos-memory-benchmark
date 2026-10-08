"""J·J&F 회복률: 영상·객체·프레임별 Native 반복 중앙값을 공통 분모로 사용.

분모 0 또는 누락은 None. 음수/100 초과를 자르거나 작은 상수를 더하지 않는다.
구간 요약은 프레임별 비율의 평균이며, 원점수 평균끼리 나누지 않는다.
"""

from collections import defaultdict
from statistics import median, variance

import settings
from evaluation.scoring.main_metrics import mean


def ratio(value, native):
    return None if value is None or native is None or native <= 0 else 100 * value / native


def point(frame, score, native, switch_frame, post=True):
    native_j = native.j if native is not None else None
    native_jf = native.jf if native is not None else None
    return {'frame': frame, 'frames_after_switch': frame - switch_frame,
            'j': score.j, 'f': score.f, 'jf': score.jf,
            'native_j': native_j, 'native_jf': native_jf,
            'recovery_j': ratio(score.j, native_j) if post else None,
            'recovery_jf': ratio(score.jf, native_jf) if post else None}


def columns(points):
    result = {}
    for metric in ('j', 'jf'):
        key = f'recovery_{metric}'
        result[key] = mean(p[key] for p in points)
        result[f'{key}_n_frames'] = sum(p[key] is not None for p in points)
        result[f'zero_native_{metric}_n_frames'] = sum(p[f'native_{metric}'] == 0 for p in points)
        result[f'missing_native_{metric}_n_frames'] = sum(p[f'native_{metric}'] is None for p in points)
    return result


def raw_point(frame, score, native, switch_frame):
    """추론 시점: 원점수만 확정한다. 반복 기준과 회복률은 집계 때 계산한다."""
    return {'frame': frame, 'frames_after_switch': frame - switch_frame,
            'j': score.j, 'f': score.f, 'jf': score.jf,
            'native_run_j': native.j if native is not None else None,
            'native_run_jf': native.jf if native is not None else None,
            'native_j': None, 'native_jf': None,
            'recovery_j': None, 'recovery_jf': None,
            'native_reference_count': 0, 'native_reference_ready': False}


def case_key(row):
    # Native 실행은 전환 조건에 공유되지만 서로 다른 객체 구간은 합치지 않는다.
    return (row['dataset'], row.get('seed', settings.EVALUATION_SEED), row['video'],
            row['object'], row['start'], row['end'])


def native_index(rows, run_ids):
    """원본에 보존한 회차별 Native를 모은다. 같은 회차의 중복 기록은 한 표본이다."""
    index = defaultdict(dict)
    for row in rows:
        run_id = row['run_id']
        if run_id not in run_ids:
            continue
        key = case_key(row)
        for p in row.get('pre_switch_frame_scores', []) + row['frame_scores']:
            # 이전 paired-run 형식도 원점수를 보존했으므로 추론 없이 재집계할 수 있다.
            value = (p.get('native_run_j', p.get('native_j')),
                     p.get('native_run_jf', p.get('native_jf')),
                     row['native_reference_id'])
            previous = index[(key, p['frame'])].get(run_id)
            if previous is not None and previous != value:
                raise ValueError(f'동일 회차/객체/프레임의 Native 원점수가 다릅니다: {key}, {run_id}, {p["frame"]}')
            index[(key, p['frame'])][run_id] = value
    return index


REFERENCE_FIELDS = ['dataset', 'seed', 'video', 'object', 'start', 'end', 'frame',
                    'native_statistic', 'native_reference_count', 'native_expected_runs', 'native_reference_ready'] + [
    f'{metric}_{name}' for metric in ('j', 'jf')
    for name in ('reference', 'mean', 'median', 'std', 'variance', 'run_count')
] + ['j_failure_count']


def reference_points(index, run_ids, statistic):
    if statistic not in ('median', 'mean'):
        raise ValueError('Native 기준은 median 또는 mean이어야 합니다.')
    # 현재는 반복 실패의 발생 빈도를 모르므로 Native 반복 중앙값을 기본 기준으로 쓴다.
    # 추후 반복 측정에서 Native 실패가 연속해서 빈번히 나타나면, 실패 사례를
    # 제외한 표를 원점수로 재집계하는 방안을 검토한다. 현재 자동 제외는 하지 않는다.
    # 제외 기준/단위와 제외 건수를 명시하고, 모든 방법에 같은 사례 선택을 적용하며
    # 원본과 실패 포함 결과를 보존한다. 제외 분석은 추론 없이 나중에 처리할 수 있다.
    reduce = median if statistic == 'median' else mean
    refs = {}
    for (key, frame), runs in sorted(index.items()):
        chosen = {r: runs[r] for r in run_ids if r in runs}
        ready = all(r in chosen and chosen[r][0] is not None and chosen[r][1] is not None for r in run_ids)
        ref = dict(zip(('dataset', 'seed', 'video', 'object', 'start', 'end'), key))
        ref.update(frame=frame, native_statistic=statistic,
                   native_reference_count=len(chosen), native_expected_runs=len(run_ids),
                   native_reference_ready=ready)
        for i, metric in enumerate(('j', 'jf')):
            values = [v[i] for v in chosen.values() if v[i] is not None]
            var = variance(values) if len(values) > 1 else None
            ref.update({f'{metric}_reference': reduce(values) if ready else None,
                        f'{metric}_mean': mean(values), f'{metric}_median': median(values) if values else None,
                        f'{metric}_std': var ** .5 if var is not None else None,
                        f'{metric}_variance': var, f'{metric}_run_count': len(values)})
        ref['j_failure_count'] = sum(v[0] is not None and v[0] <= settings.RECALL_J for v in chosen.values())
        refs[(key, frame)] = ref
    return refs


def recompute(rows, run_ids=None, statistic='median'):
    """새 결과를 반환한다. 입력 원점수/파일은 바꾸지 않는다. 실패 0점도 중앙값 표본에 포함."""
    run_ids = tuple(run_ids if run_ids is not None else range(1, settings.EVALUATION_RUNS + 1))
    if not run_ids or len(set(run_ids)) != len(run_ids) or any(r < 1 for r in run_ids):
        raise ValueError('서로 다른 양수 회차 번호가 필요합니다.')
    rows = [r for r in rows if r['run_id'] in run_ids]
    refs = reference_points(native_index(rows, run_ids), run_ids, statistic)
    out = []
    for row in rows:
        result = dict(row)
        result.update(recovery_reference=f'native_{statistic}', recovery_reference_runs=list(run_ids))
        for field in ('frame_scores', 'pre_switch_frame_scores'):
            points = []
            for original in row.get(field, []):
                p = dict(original)
                p['native_run_j'] = original.get('native_run_j', original.get('native_j'))
                p['native_run_jf'] = original.get('native_run_jf', original.get('native_jf'))
                ref = refs[(case_key(row), p['frame'])]
                p.update(native_j=ref['j_reference'], native_jf=ref['jf_reference'],
                         native_reference_count=ref['native_reference_count'],
                         native_reference_ready=ref['native_reference_ready'])
                for metric in ('j', 'jf'):
                    p[f'recovery_{metric}'] = ratio(p[metric], p[f'native_{metric}']) if field == 'frame_scores' else None
                points.append(p)
            result[field] = points
        result.update(columns(result['frame_scores']))
        result['pending_native_n_frames'] = sum(not p['native_reference_ready'] for p in result['frame_scores'])
        out.append(result)
    return out


def reference_rows(rows, run_ids=None, statistic='median'):
    run_ids = tuple(run_ids if run_ids is not None else range(1, settings.EVALUATION_RUNS + 1))
    return list(reference_points(native_index(rows, run_ids), run_ids, statistic).values())
