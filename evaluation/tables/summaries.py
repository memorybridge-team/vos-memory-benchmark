"""동일 사례를 회차마다 집계한 뒤 반복 평균·표본 분산(ddof=1)을 구한다.

회차별: 프레임 평균(결과 줄) → 객체 평균 → 영상 평균.
미완료/분모 0로 값이 빠진 경우 회차 간 공통 유효 사례만 사용하고 수를 보고한다.
회차별 표준편차를 평균하거나 서로 다른 영상의 점수를 반복 표본으로 세지 않는다.
"""

from collections import defaultdict
from statistics import variance

from evaluation.scoring.main_metrics import mean

METRICS = {'j': 100, 'jf': 100, 'recovery_j': 1, 'recovery_jf': 1,
           'failure_rate': 100, 'switch_seconds': 1, 'switch_gpu_mb': 1}


def distribution(values):
    values = [v for v in values if v is not None]
    var = variance(values) if len(values) > 1 else None
    return {'mean': mean(values), 'std': var ** 0.5 if var is not None else None,
            'variance': var, 'run_count': len(values)}


def case(row):
    return (row['dataset'], row['video'], row.get('object', 0),
            row['switch_name'], row.get('switch_frame', 0))


def metric_stats(rows, metric, scale=1):
    by_run = defaultdict(dict)
    for row in rows:
        by_run[row.get('run_id', 1)][case(row)] = row
    valid = [{key for key, row in entries.items() if row.get(metric) is not None}
             for entries in by_run.values()]
    common = set.intersection(*valid) if valid else set()
    per_run = {}
    for run_id, entries in sorted(by_run.items()):
        videos = defaultdict(list)
        for key in sorted(common):
            row = entries[key]
            videos[(row['dataset'], row['video'])].append(row[metric] * scale)
        per_run[run_id] = mean(mean(values) for values in videos.values())
    result = distribution(per_run.values())
    result.update(per_run=per_run, object_count=len(common),
                  video_count=len({key[:2] for key in common}),
                  available_run_count=len(by_run))
    return result


def format_stats(rows, metric, scale=1, digits=1):
    stats = metric_stats(rows, metric, scale)
    if stats['mean'] is None:
        return '-'
    text = f'{stats["mean"]:.{digits}f}'
    return text + (f' ± {stats["std"]:.{digits}f}' if stats['std'] is not None else ' (n=1)')


GROUP_FIELDS = ['dataset', 'switch_name', 'baseline']
SUMMARY_FIELDS = GROUP_FIELDS + [f'{m}_{s}' for m in METRICS for s in ('mean', 'std', 'variance', 'run_count')] + ['video_count', 'object_count', 'recovery_j_video_count', 'recovery_j_object_count', 'recovery_jf_video_count', 'recovery_jf_object_count']
PER_RUN_FIELDS = GROUP_FIELDS + ['run_id'] + list(METRICS) + ['video_count', 'object_count']
PER_VIDEO_FIELDS = GROUP_FIELDS + ['run_id', 'video'] + list(METRICS) + ['object_count', 'n_frames', 'recovery_j_n_frames', 'recovery_jf_n_frames', 'zero_native_j_n_frames', 'zero_native_jf_n_frames', 'pending_native_n_frames']


def build(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP_FIELDS)].append(row)
    summaries, per_runs, per_videos = [], [], []
    for key, chosen in sorted(groups.items()):
        identity = dict(zip(GROUP_FIELDS, key))
        summary = dict(identity)
        all_stats = {m: metric_stats(chosen, m, scale) for m, scale in METRICS.items()}
        for metric, stats in all_stats.items():
            for name in ('mean', 'std', 'variance', 'run_count'):
                summary[f'{metric}_{name}'] = stats[name]
        summary.update(video_count=all_stats['j']['video_count'], object_count=all_stats['j']['object_count'])
        for m in ('recovery_j', 'recovery_jf'):
            summary[f'{m}_video_count'] = all_stats[m]['video_count']
            summary[f'{m}_object_count'] = all_stats[m]['object_count']
        summaries.append(summary)
        for run_id in sorted({r['run_id'] for r in chosen}):
            per_runs.append({**identity, 'run_id': run_id,
                             **{m: stats['per_run'][run_id] for m, stats in all_stats.items()},
                             'video_count': summary['video_count'], 'object_count': summary['object_count']})
        videos = defaultdict(list)
        for row in chosen:
            videos[(row['run_id'], row['video'])].append(row)
        for (run_id, video), objects in sorted(videos.items()):
            out = {**identity, 'run_id': run_id, 'video': video, 'object_count': len(objects)}
            for metric, scale in METRICS.items():
                value = mean(r.get(metric) for r in objects)
                out[metric] = value * scale if value is not None else None
            for count in ('n_frames', 'recovery_j_n_frames', 'recovery_jf_n_frames', 'zero_native_j_n_frames', 'zero_native_jf_n_frames', 'pending_native_n_frames'):
                out[count] = sum(r.get(count, 0) for r in objects)
            per_videos.append(out)
    return summaries, per_runs, per_videos
