"""프레임별 비율, Native 반복 중앙값 분모, 0점/0분모, 반복 통계의 실제 수식을 검증한다."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.scoring.jf import FrameScore
from evaluation.scoring import recovery
from evaluation.tables import summaries, temporal


def close(a, b):
    assert abs(a - b) < 1e-10, (a, b)


def test_recovery():
    # 평균의 비율(90%) 대신 비율의 평균(75%). J와 J&F 모두 독립 계산한다.
    points = [recovery.point(11, FrameScore(.1, .1, True), FrameScore(.2, .2, True), 10),
              recovery.point(12, FrameScore(.8, .8, True), FrameScore(.8, .8, True), 10)]
    cols = recovery.columns(points)
    close(cols['recovery_j'], 75)
    close(cols['recovery_jf'], 75)
    assert cols['recovery_j_n_frames'] == cols['recovery_jf_n_frames'] == 2
    assert recovery.ratio(0, .8) == 0, '성능 실패 0점은 유효 회복률이다'
    assert recovery.ratio(.9, .5) == 180, '100% 초과를 자르지 않는다'
    assert recovery.ratio(0, 0) is None
    assert recovery.ratio(.5, None) is None
    # J만 0인 Native에서 J&F 분모는 양수일 수 있다.
    point = recovery.point(11, FrameScore(.2, .4, True), FrameScore(0, .8, True), 10)
    assert point['recovery_j'] is None
    close(point['recovery_jf'], 75)
    cols = recovery.columns([point])
    assert cols['zero_native_j_n_frames'] == 1 and cols['zero_native_jf_n_frames'] == 0
    assert cols['recovery_j_n_frames'] == 0 and cols['recovery_jf_n_frames'] == 1
    # 비율 계산의 기본 수식. 실제 분모는 Native 반복 중앙값이다.
    close(recovery.ratio(.4, .5), 80)
    close(recovery.ratio(.4, .8), 50)

    rows = []
    for run_id, value in enumerate((.8, .8, 0), 1):
        # 영상 a의 객체 2개와 영상 b의 객체 1개: 객체 수와 무관하게 영상에 동일 가중치.
        for video, obj, score in (('a', 1, value), ('a', 2, value), ('b', 1, value / 2)):
            rows.append({'dataset': 'm3vos', 'video': video, 'object': obj, 'switch_name': '50',
                         'switch_frame': 10, 'baseline': 'source_only', 'run_id': run_id,
                         'j': score, 'jf': score, 'recovery_j': score * 100,
                         'recovery_jf': score * 100, 'failure_rate': 1 if score == 0 else 0,
                         'frame_scores': [{'frame': 11, 'frames_after_switch': 1, 'j': score, 'jf': score,
                                           'native_j': 1, 'native_jf': 1,
                                           'recovery_j': score * 100, 'recovery_jf': score * 100}]})
    stats = summaries.metric_stats(rows, 'j', 100)
    close(stats['mean'], 40)
    close(stats['std'], math.sqrt(1200))
    for run_id, expected in ((1, 60), (2, 60), (3, 0)):
        close(stats['per_run'][run_id], expected)
    assert stats['run_count'] == 3 and stats['video_count'] == 2 and stats['object_count'] == 3
    assert summaries.metric_stats(rows, 'failure_rate', 100)['mean'] == 100 / 3
    point = temporal.build(rows)[0]
    close(point['j'], stats['mean'])
    close(point['j_std'], stats['std'])
    close(point['recovery_j'], 40)
    assert point['recovery_j_run_count'] == 3 and point['object_count'] == 3
    # 같은 평가 대상의 80,80,0은 평균53.3. 실패를 빼지 않는다.
    dist = summaries.distribution([80, 80, 0])
    close(dist['mean'], 160 / 3)
    assert summaries.distribution([80])['std'] is None
    assert summaries.distribution([80])['variance'] is None
    assert summaries.distribution([])['mean'] is None
    # 부분 완료 회차를 서로 다른 대상으로 평균하지 않는다.
    partial = [r for r in rows if not (r['run_id'] == 2 and r['video'] == 'b')]
    stats = summaries.metric_stats(partial, 'j', 100)
    assert stats['video_count'] == 1 and stats['object_count'] == 2
    close(stats['mean'], 160 / 3)
    # Native 분모0은 회복률만 N/A, 원점수는 유지한다.
    broken = [dict(r) for r in rows]
    broken[0]['recovery_j'] = None
    stats = summaries.metric_stats(broken, 'recovery_j')
    assert stats['object_count'] == 2
    assert summaries.metric_stats(broken, 'j')['object_count'] == 3
    # 추론 원점수를 보존하고 같은 프레임 Native 3회 중앙값으로 재집계한다.
    import json
    from types import SimpleNamespace
    from evaluation.evaluate_video import Run, _row
    video = SimpleNamespace(dataset='m3vos', name='a')
    obj = {'object': 1, 'start': 0, 'end': 11}
    sw = {'name': '50', 'frame': 10}
    method = SimpleNamespace(name='source_only', role='main', revision=1)
    base = SimpleNamespace(name='full_replay', role='main', revision=1)
    run = Run(scores={0: FrameScore(1, 1, True), 11: FrameScore(.72, .72, True)})
    cost = {'switch_seconds': None, 'switch_gpu_mb': None}
    raw = []
    for run_id, native_score in enumerate((.8, .8, 0), 1):
        reference = {'native_reference_id': str(run_id),
                     'scores': [{'frame': 0, 'j': 1, 'f': 1, 'gt_visible': True},
                                {'frame': 11, 'j': native_score, 'f': native_score, 'gt_visible': True}]}
        row = _row(video, obj, sw, method, run, cost, reference, run, run_id, 0)
        assert row['recovery_j'] is row['recovery_jf'] is None
        assert row['frame_scores'][0]['native_run_j'] == native_score
        raw.append(row)
        native_run = Run(scores={0: FrameScore(1, 1, True),
                                 11: FrameScore(native_score, native_score, True)})
        raw.append(_row(video, obj, sw, base, native_run, cost, reference, native_run, run_id, 0))
    original = json.dumps(raw, sort_keys=True)
    med = recovery.recompute(raw)
    avg = recovery.recompute(raw, statistic='mean')
    assert json.dumps(raw, sort_keys=True) == original
    assert recovery.recompute(med, statistic='mean') == avg, '정의 변경 시 집계된 Native 대신 원점수를 써야 한다'
    for row in med:
        assert row['frame_scores'][0]['native_j'] == .8
        assert row['frame_scores'][0]['native_reference_count'] == 3
        assert row['recovery_reference_runs'] == [1, 2, 3]
        if row['baseline'] == 'source_only':
            close(row['recovery_j'], 90)
            close(row['recovery_jf'], 90)
    for row in avg:
        if row['baseline'] == 'source_only':
            close(row['recovery_j'], 135)
    native_med = [r for r in med if r['baseline'] == 'full_replay']
    close(summaries.metric_stats(native_med, 'recovery_j')['mean'], 200 / 3)
    refs = recovery.reference_rows(raw)
    frame = next(p for p in refs if p['frame'] == 11)
    assert frame['j_failure_count'] == 1
    close(frame['j_median'], .8)
    close(frame['j_mean'], 1.6 / 3)
    assert frame['native_reference_count'] == 3, '방법数・切断数でNative試行を重複して数えない'
    # 2회만 완료되었으면 3회 중앙값을 확정하지 않는다.
    pending = recovery.recompute([r for r in raw if r['run_id'] != 3])
    assert all(r['recovery_j'] is None and r['pending_native_n_frames'] == 1 for r in pending)
    assert all(r['frame_scores'][0]['native_reference_count'] == 2 for r in pending)
    # 명시적으로 1회 분석을 선택한 경우에는 그 1회가 기준이다.
    one = recovery.recompute(raw, run_ids=[1])
    assert all(r['frame_scores'][0]['native_reference_count'] == 1 for r in one)
    # Native 모두 실패/과반수 실패: 중앙값0이면 회복률만 N/A. 실패값을 삭제하지 않는다.
    for native_values in ((0, 0, 0), (0, 0, .8)):
        changed = []
        for row in raw:
            copied = dict(row, frame_scores=[dict(p) for p in row['frame_scores']])
            p = copied['frame_scores'][0]
            p['native_run_j'] = p['native_run_jf'] = native_values[row['run_id'] - 1]
            changed.append(copied)
        result = recovery.recompute(changed)
        assert all(r['recovery_j'] is r['recovery_jf'] is None for r in result)
        assert all(r['zero_native_j_n_frames'] == 1 and r['pending_native_n_frames'] == 0 for r in result)
    # J&F 중앙값은 J 중앙값과 F 중앙값의 평균이 아니다.
    changed = []
    for row in raw:
        copied = dict(row, frame_scores=[dict(p) for p in row['frame_scores']])
        p = copied['frame_scores'][0]
        j, f = ((0, 1), (.4, .8), (1, 0))[row['run_id'] - 1]
        p['native_run_j'], p['native_run_jf'] = j, (j + f) / 2
        changed.append(copied)
    result = recovery.recompute(changed)
    assert all(r['frame_scores'][0]['native_j'] == .4 and r['frame_scores'][0]['native_jf'] == .5 for r in result)
    # 같은 회차의 Native 기록이 모순되면 섞지 않는다.
    mismatch = [dict(row, frame_scores=[dict(p) for p in row['frame_scores']]) for row in raw]
    mismatch[0]['frame_scores'][0]['native_run_j'] = .7
    try:
        recovery.recompute(mismatch)
        raise AssertionError('서로 다른 Native 기록을 허용함')
    except ValueError:
        pass
    print('OK: J·J&F 프레임 회복률, 0분모, 실패 포함, 영상 가중치, 반복 분산, 공통 표본')


if __name__ == '__main__':
    test_recovery()
