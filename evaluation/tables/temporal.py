"""전환 전후 전체 곡선. 시간점별 객체 → 영상 평균 후 회차 평균·표본 분산."""

# 최단 구간 [-n,+n] 그림은 recovery_curves.py에서 별도로 만든다.
# TODO: 전체 원점수 기반 최종 논문 표현은 추후 재검토한다.
# 현재 CSV는 중간 집계이며, 원본 프레임별 값으로 언제든 다시 표현할 수 있다.

from collections import defaultdict

from evaluation.methods import METHODS
from evaluation.tables.summaries import metric_stats

METRICS = {'j': 100, 'jf': 100, 'native_j': 100, 'native_jf': 100,
           'native_run_j': 100, 'native_run_jf': 100,
           'recovery_j': 1, 'recovery_jf': 1}
CSV_FIELDS = ['dataset', 'switch_name', 'baseline', 'phase', 'frames_after_switch'] + [
    name for m in METRICS for name in (m, f'{m}_std', f'{m}_variance', f'{m}_run_count')
] + ['video_count', 'object_count', 'recovery_j_object_count', 'recovery_jf_object_count']


def build(rows):
    values = defaultdict(list)
    for index, row in enumerate(rows):
        points = row.get('pre_switch_frame_scores', []) + row['frame_scores']
        for point in points:
            key = (row['dataset'], row['switch_name'], row['baseline'], point['frames_after_switch'])
            values[key].append({
                'dataset': row['dataset'], 'video': row['video'],
                'object': row.get('object', index), 'run_id': row.get('run_id', 1),
                'switch_name': row['switch_name'], 'switch_frame': row.get('switch_frame', 0),
                **point})
    order = {m.name: i for i, m in enumerate(METHODS)}
    out = []
    for key in sorted(values, key=lambda k: (k[0], int(k[1]), order[k[2]], k[3])):
        stats = {m: metric_stats(values[key], m, scale) for m, scale in METRICS.items()}
        result = {'dataset': key[0], 'switch_name': key[1], 'baseline': key[2],
                  'phase': 'post' if key[3] > 0 else 'pre', 'frames_after_switch': key[3],
                  'video_count': stats['j']['video_count'], 'object_count': stats['j']['object_count'],
                  'recovery_j_object_count': stats['recovery_j']['object_count'],
                  'recovery_jf_object_count': stats['recovery_jf']['object_count']}
        for metric, item in stats.items():
            result[metric] = item['mean']
            for name in ('std', 'variance', 'run_count'):
                result[f'{metric}_{name}'] = item[name]
        out.append(result)
    return out
