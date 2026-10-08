"""최단 관찰 구간에 맞춘 전환 전후 프레임별 평균 회복률.

n = min(s-start, end-s), 데이터셋/전환 비율별 모든 계획 객체에 대한 최솟값.
평가 단위는 영상 전체가 아니라 객체 최초 등장~영상 끝이므로 늦게 등장한
객체가 더 짧은 관찰 구간을 만들 수 있다. 0=s는 마지막 Small 프레임이다.

한계(최종 논문 표현은 추후 재검토):
- 최단 구간 하나가 n을 결정하며, 긴 영상의 [-n,+n] 밖은 그림에서 잘린다.
  장기 추적의 악화/회복이나 뒤늦은 실패를 이 그림으로 설명할 수 없다.
- 같은 프레임 수가 같은 실제 시간/FPS, 사건, 난이도를 뜻하지 않는다.
- 공통 구간은 길이에 따른 중도 탈락만 막는다. GT 가려짐/누락 및 Native=0
  때문에 점수가 없는 사례가 생겨 시간점별 유효 표본 수는 여전히 달라진다.
  보간/0점 대체/실패 사례 자동 제외 없이 N(t)를 CSV와 그림에 표시한다.
- 모든 방법/회차가 완료된 객체만 그림의 공통 모집단에 포함한다. 기술적
  미완료 사례는 모든 방법에서 같이 제외하고 목록을 저장한다. 점수 실패는
  그대로 포함한다. 원본 전체 프레임 및 전체 구간 평가표는 자르지 않는다.
- 곡선은 프레임별 비율의 평균이다. 평가표의 구간 평균 점수 비율과 다르다.
"""

from collections import defaultdict
from pathlib import Path

import settings
from evaluation.methods import METHODS
from evaluation.tables.summaries import metric_stats

METRICS = ('recovery_j', 'recovery_jf')
CSV_FIELDS = ['dataset', 'switch_name', 'baseline', 'frames_after_switch', 'phase', 'window_n',
              'cohort_video_count', 'cohort_object_count'] + [
    f'{metric}{suffix}' for metric in METRICS
    for suffix in ('', '_std', '_variance', '_run_count', '_video_count', '_object_count')
]
CASE_FIELDS = ('video', 'object', 'start', 'end', 'switch_frame')


def _case(row):
    return tuple(row[key] for key in CASE_FIELDS)


def planned_cases(video_list, switch_name):
    return { (video['video'], obj['object'], obj['start'], obj['end'], sw['frame'])
             for video in video_list['videos'] for obj in video['objects']
             for sw in obj['switches'] if sw['name'] == switch_name }


def build(rows, video_lists=None, run_ids=None, methods=None):
    """새 집계만 반환하며 원점수와 전체 구간 표는 바꾸지 않는다."""
    run_ids = tuple(run_ids if run_ids is not None else range(1, settings.EVALUATION_RUNS + 1))
    methods = tuple(methods if methods is not None else (m.name for m in METHODS))
    if not run_ids or len(set(run_ids)) != len(run_ids) or any(r < 1 for r in run_ids):
        raise ValueError('곡선에는 서로 다른 회차 번호가 필요합니다.')
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('곡선에는 서로 다른 방법 이름이 필요합니다.')
    groups = defaultdict(list)
    for row in rows:
        if row['run_id'] in run_ids and row['baseline'] in methods:
            groups[(row['dataset'], row['switch_name'])].append(row)
    curves, windows = [], []
    for (dataset, switch_name), chosen in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        recorded = {_case(row) for row in chosen}
        video_list = (video_lists or {}).get(dataset)
        planned = planned_cases(video_list, switch_name) if video_list is not None else recorded
        if not recorded.issubset(planned):
            raise ValueError(f'{dataset}/{switch_name}: 결과와 계획 영상·객체 구간이 다릅니다.')
        if not planned:
            continue
        capacities = {case: min(case[4] - case[2], case[3] - case[4]) for case in planned}
        n = min(capacities.values())
        if n < 0:
            raise ValueError('전환 프레임이 객체 추적 구간 밖에 있습니다.')
        index = {}
        for row in chosen:
            key = (_case(row), row['baseline'], row['run_id'])
            if key in index and index[key] != row:
                raise ValueError(f'동일 그래프 조건의 결과가 다릅니다: {key}')
            index[key] = row
        complete, excluded = [], []
        for case in sorted(planned):
            missing = [{'baseline': method, 'run_id': run_id} for method in methods for run_id in run_ids
                       if (case, method, run_id) not in index]
            if missing:
                excluded.append(dict(zip(CASE_FIELDS, case), missing_conditions=missing))
            else:
                complete.append(case)
        window = dict(dataset=dataset, switch_name=switch_name, window_n=n,
                      relative_frame_min=-n, relative_frame_max=n,
                      zero_definition='last_pre_switch_frame',
                      window_basis='planned_object_ranges' if video_list is not None else 'recorded_object_ranges',
                      run_ids=list(run_ids), methods=list(methods),
                      planned_video_count=len({case[0] for case in planned}), planned_object_count=len(planned),
                      cohort_video_count=len({case[0] for case in complete}), cohort_object_count=len(complete),
                      shortest_video_frames=min(case[3] + 1 for case in planned),
                      shortest_object_span_frames=min(case[3] - case[2] + 1 for case in planned),
                      limiting_cases=[dict(zip(CASE_FIELDS, c)) for c in sorted(planned) if capacities[c] == n],
                      cohort_cases=[dict(zip(CASE_FIELDS, c)) for c in complete],
                      excluded_incomplete_cases=excluded,
                      cropped_pre_frames_per_object_sum=sum(c[4] - c[2] - n for c in complete),
                      cropped_post_frames_per_object_sum=sum(c[3] - c[4] - n for c in complete),
                      status='ok' if complete else 'no_complete_cases')
        windows.append(window)
        point_maps = {}
        for case in complete:
            for method in methods:
                for run_id in run_ids:
                    row = index[(case, method, run_id)]
                    points = row.get('pre_switch_frame_scores', []) + row['frame_scores']
                    point_maps[(case, method, run_id)] = {
                        p['frames_after_switch']: p for p in points if -n <= p['frames_after_switch'] <= n
                    }
        for method in methods:
            for t in range(-n, n + 1):
                values = []
                for case in complete:
                    for run_id in run_ids:
                        # 누락을 명시하고 같은 시간점의 회차 간 공통 유효 사례를 집계한다.
                        point = point_maps[(case, method, run_id)].get(t, {})
                        values.append({'dataset': dataset, 'video': case[0], 'object': case[1],
                                       'switch_name': switch_name, 'switch_frame': case[4], 'run_id': run_id,
                                       **{metric: point.get(metric) for metric in METRICS}})
                result = {'dataset': dataset, 'switch_name': switch_name, 'baseline': method,
                          'frames_after_switch': t, 'phase': 'post' if t > 0 else 'pre', 'window_n': n,
                          'cohort_video_count': window['cohort_video_count'],
                          'cohort_object_count': len(complete)}
                for metric in METRICS:
                    stats = metric_stats(values, metric)
                    result[metric] = stats['mean']
                    for key in ('std', 'variance', 'run_count', 'video_count', 'object_count'):
                        result[f'{metric}_{key}'] = stats[key]
                curves.append(result)
    return curves, windows


def plot(curves, windows, output_dir):
    """데이터셋×전환 비율별 J/J&F 두 패널에 모든 방법의 평균 곡선을 저장한다."""
    if not windows:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = {m.name: ('Translator' if m.name == 'translator' else m.label) for m in METHODS}
    styles = {'source_only': ('#636363', '-'), 'full_replay': ('#0072B2', '--'),
              'direct_state_copy': ('#D55E00', '-'), 'original_last_visible': ('#009E73', '-.'),
              'original_replay_8': ('#CC79A7', ':'), 'translator': ('#E69F00', '-')}
    groups = defaultdict(list)
    for point in curves:
        groups[(point['dataset'], point['switch_name'], point['baseline'])].append(point)
    paths = []
    for window in windows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True)
        n = window['window_n']
        for ax, metric, title in zip(axes, METRICS, ('J recovery', 'J&F recovery')):
            for method in window['methods']:
                series = groups[(window['dataset'], window['switch_name'], method)]
                # 공통 Small 전반은 Source-only 선으로 표시하고 각 방법은 t=0부터 이어 그린다.
                if method not in ('source_only', 'full_replay'):
                    series = [p for p in series if p['frames_after_switch'] >= 0]
                xs = [p['frames_after_switch'] for p in series]
                ys = [float('nan') if p[metric] is None else p[metric] for p in series]
                color, linestyle = styles.get(method, (None, '-'))
                ax.plot(xs, ys, label=labels.get(method, method), color=color, linestyle=linestyle, linewidth=1.6)
            ax.axvline(0, color='black', linestyle=':', linewidth=.9)
            ax.axhline(100, color='black', linestyle='--', linewidth=.6, alpha=.4)
            ax.set_xlim((-n, n) if n else (-.5, .5))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(title)
            ax.set_xlabel('Relative frame (0 = last pre-switch frame)')
            ax.set_ylabel('Recovery (%)')
            ax.grid(axis='y', alpha=.2)
            supports = [p[f'{metric}_video_count'] for method in window['methods']
                        for p in groups[(window['dataset'], window['switch_name'], method)]]
            object_supports = [p[f'{metric}_object_count'] for method in window['methods']
                               for p in groups[(window['dataset'], window['switch_name'], method)]]
            ax.text(.02, .02,
                    f'Valid videos: {min(supports, default=0)}–{max(supports, default=0)}; '
                    f'objects: {min(object_supports, default=0)}–{max(object_supports, default=0)}',
                    transform=ax.transAxes, fontsize=8, color='#555555')
            if window['status'] != 'ok':
                ax.text(.5, .5, 'No cases completed across all methods and runs',
                        ha='center', va='center', transform=ax.transAxes, fontsize=9)
        fig.suptitle(f"{window['dataset']} | switch {window['switch_name']}% | n={n} | "
                     f"videos={window['cohort_video_count']}/{window['planned_video_count']} | "
                     f"runs={len(window['run_ids'])}", fontsize=11)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .045), ncol=3, fontsize=8)
        fig.text(.5, .015, 'Framewise ratios; object → video → run means. Outside-window frames are omitted.',
                 ha='center', fontsize=8)
        fig.subplots_adjust(left=.07, right=.98, top=.85, bottom=.27, wspace=.25)
        stem = output_dir / f"{window['dataset']}.switch{window['switch_name']}"
        for extension in ('png', 'pdf'):
            path = stem.with_suffix(stem.suffix + '.' + extension)
            fig.savefig(path, dpi=180)
            paths.append(path)
        plt.close(fig)
    return paths
