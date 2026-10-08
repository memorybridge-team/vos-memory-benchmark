"""공통 구간 경계, 영상 가중치, 미완료/가림/0분모와 그림 저장을 검증한다."""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.tables import recovery_curves


def close(actual, expected):
    assert abs(actual - expected) < 1e-10, (actual, expected)


def fixture():
    rows = []
    methods = ('source_only', 'translator')
    cases = [('short', 1, 0, 8, 4, 0, 20), ('short', 2, 0, 8, 4, 100, 80),
             ('long', 1, 0, 20, 10, 80, 100)]
    for video, obj, start, end, switch, j, jf in cases:
        for method in methods:
            for run_id, factor in ((1, 1), (2, .5)):
                points = [{'frame': frame, 'frames_after_switch': frame - switch,
                           'recovery_j': j * factor, 'recovery_jf': jf * factor}
                          for frame in range(start, end + 1)]
                # 같은 시점의 GT 가려짐: 모든 방법/회차에 같은 누락.
                if video == 'short' and obj == 1:
                    points = [p for p in points if p['frames_after_switch'] != 1]
                rows.append({'dataset': 'demo', 'video': video, 'object': obj, 'start': start,
                             'end': end, 'switch_frame': switch, 'switch_name': '50',
                             'baseline': method, 'run_id': run_id,
                             'pre_switch_frame_scores': [p for p in points if p['frame'] <= switch],
                             'frame_scores': [p for p in points if p['frame'] > switch]})
    return rows, methods


def get(curves, t, method='source_only'):
    return next(p for p in curves if p['frames_after_switch'] == t and p['baseline'] == method)


def test_window():
    rows, methods = fixture()
    before = json.dumps(rows, sort_keys=True)
    curves, windows = recovery_curves.build(rows, run_ids=(1, 2), methods=methods)
    assert json.dumps(rows, sort_keys=True) == before
    window = windows[0]
    assert window['window_n'] == 4
    assert window['cohort_video_count'] == 2 and window['cohort_object_count'] == 3
    assert window['shortest_video_frames'] == 9
    assert window['cropped_pre_frames_per_object_sum'] == window['cropped_post_frames_per_object_sum'] == 6
    assert len(curves) == 9 * 2
    assert {p['frames_after_switch'] for p in curves} == set(range(-4, 5))
    assert get(curves, 0)['phase'] == 'pre' and get(curves, 1)['phase'] == 'post'
    # short의 두 객체를 먼저 평균한 뒤 long과 평균한다. 영상에 같은 가중치를 준다.
    point = get(curves, -1)
    close(point['recovery_j'], 48.75)
    close(point['recovery_jf'], 56.25)
    close(point['recovery_j_std'], math.sqrt((65 - 32.5) ** 2 / 2))
    assert point['recovery_j_video_count'] == 2 and point['recovery_j_object_count'] == 3
    # 구간을 고정해도 유효 표본 수는 달라질 수 있다. 가려진 객체의 값을 채우지 않는다.
    point = get(curves, 1)
    close(point['recovery_j'], 67.5)
    assert point['recovery_j_video_count'] == 2 and point['recovery_j_object_count'] == 2
    assert point['cohort_object_count'] == 3
    # Native 분모0으로 전체가 N/A인 시점도 위치와 빈 값을 보존한다.
    for row in rows:
        p = next(p for p in row['frame_scores'] if p['frames_after_switch'] == 2)
        p['recovery_j'] = p['recovery_jf'] = None
    curves, _ = recovery_curves.build(rows, run_ids=(1, 2), methods=methods)
    assert get(curves, 2)['recovery_j'] is None and get(curves, 2)['recovery_j_video_count'] == 0
    assert get(curves, 2)['cohort_object_count'] == 3
    assert get(curves, 2)['recovery_j_run_count'] == 0


def test_plan_and_completion():
    rows, methods = fixture()
    # 짧은 객체가 미완료면 모든 방법에서 같이 제외한다. n은 전체 계획으로 정한다.
    partial = [r for r in rows if not (r['video'] == 'short' and r['baseline'] == 'translator' and r['run_id'] == 2)]
    curves, windows = recovery_curves.build(partial, run_ids=(1, 2), methods=methods)
    assert windows[0]['window_n'] == 4 and windows[0]['cohort_video_count'] == 1
    assert len(windows[0]['excluded_incomplete_cases']) == 2
    assert all(p['cohort_object_count'] == 1 for p in curves)
    # 계획의 더 짧은 객체가 아직 실행되지 않아도 완료된 긴 영상으로 n을 늘리지 않는다.
    plan = {'demo': {'videos': [
        {'video': 'short', 'objects': [{'object': i, 'start': 0, 'end': 8, 'switches': [{'name': '50', 'frame': 4}]} for i in (1, 2)]},
        {'video': 'long', 'objects': [{'object': 1, 'start': 0, 'end': 20, 'switches': [{'name': '50', 'frame': 10}]}]},
        {'video': 'late', 'objects': [{'object': 1, 'start': 14, 'end': 20, 'switches': [{'name': '50', 'frame': 17}]}]},
    ]}}
    curves, windows = recovery_curves.build(rows, plan, run_ids=(1, 2), methods=methods)
    assert windows[0]['window_n'] == 3
    assert windows[0]['shortest_video_frames'] == 9 and windows[0]['shortest_object_span_frames'] == 7
    assert windows[0]['window_basis'] == 'planned_object_ranges'
    assert windows[0]['planned_video_count'] == 3 and windows[0]['cohort_video_count'] == 2
    assert windows[0]['limiting_cases'][0]['video'] == 'late'
    assert len(curves) == 7 * 2
    # 완료된 방법이 없으면 다른 방법만의 표본으로 비교 곡선을 만들지 않는다.
    empty, meta = recovery_curves.build(rows, run_ids=(1, 2), methods=(*methods, 'full_replay'))
    assert meta[0]['status'] == 'no_complete_cases'
    assert all(p['recovery_j'] is None for p in empty)
    assert recovery_curves.build([]) == ([], [])


def test_cut_groups_and_plot():
    rows, methods = fixture()
    later = []
    for row in rows:
        switch = round(.75 * row['end'])
        points = row['pre_switch_frame_scores'] + row['frame_scores']
        shifted = [dict(p, frames_after_switch=p['frame'] - switch) for p in points]
        later.append(dict(row, switch_name='75', switch_frame=switch,
                          pre_switch_frame_scores=[p for p in shifted if p['frame'] <= switch],
                          frame_scores=[p for p in shifted if p['frame'] > switch]))
    curves, windows = recovery_curves.build(rows + later, run_ids=(1, 2), methods=methods)
    assert [(w['switch_name'], w['window_n']) for w in windows] == [('50', 4), ('75', 2)]
    with tempfile.TemporaryDirectory() as folder:
        paths = recovery_curves.plot(curves, windows, folder)
        assert len(paths) == 4
        assert all(p.stat().st_size > 1000 for p in paths)
        assert {p.suffix for p in paths} == {'.png', '.pdf'}
        assert {p.name for p in paths} == {'demo.switch50.png', 'demo.switch50.pdf', 'demo.switch75.png', 'demo.switch75.pdf'}


if __name__ == '__main__':
    test_window()
    test_plan_and_completion()
    test_cut_groups_and_plot()
    print('OK: 최단 구간, 전환별 n, 공통 사례, 영상 가중치, 누락/실패, PNG/PDF')
