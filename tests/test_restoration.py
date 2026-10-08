"""프레임별 R²의 수학, 누락/퇴화 조건 및 s+1 이전 측정·비용 제외 검증."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import cost
from evaluation.evaluate_video import _run_base
from evaluation.scoring import restoration


def close(actual, expected):
    assert abs(actual - expected) < 1e-12, (actual, expected)


def test_math():
    y = torch.tensor([1., 2., 3.])
    original = y.clone()
    for prediction, expected in (([1., 2., 3.], 1), ([1., 2., 2.], .5),
                                 ([2., 2., 2.], 0), ([3., 2., 1.], -3),
                                 ([2., 3., 4.], -.5)):
        score = restoration.tensor_score(y, torch.tensor(prediction))
        assert score['status'] == 'ok'
        close(score['r2'], expected)
        close(score['sst'], 2)
        assert score['native_mean'] == 2 and score['n_elements'] == 3
    assert torch.equal(y, original)
    # BF16도 float64로 변환해 계산하고 입력 dtype을 바꾸지 않는다.
    close(restoration.tensor_score(y.to(torch.bfloat16), y)['r2'], 1)
    for prediction in (torch.ones(3), torch.zeros(3)):
        score = restoration.tensor_score(torch.ones(3), prediction)
        assert score['r2'] is None and score['status'] == 'zero_native_variance'
        assert score['sst'] == 0 and score['sse'] is not None
    assert restoration.tensor_score(torch.ones(1), torch.ones(1))['status'] == 'insufficient_elements'
    assert restoration.tensor_score(torch.ones(0), torch.ones(0))['status'] == 'empty_tensor'
    assert restoration.tensor_score(y, y.reshape(1, 3))['status'] == 'shape_mismatch'
    assert restoration.tensor_score(None, y)['status'] == 'missing_native_tensor'
    assert restoration.tensor_score(y, None)['status'] == 'missing_target_tensor'
    assert restoration.tensor_score(y, torch.tensor([1., float('nan'), 3.]))['status'] == 'nonfinite_tensor'
    assert restoration.tensor_score(torch.tensor([1., float('inf'), 3.]), y)['status'] == 'nonfinite_tensor'


def test_memory_frames():
    y = torch.tensor([1., 2., 3.])
    native = {0: {'maskmem_features': y, 'obj_ptr': y, 'is_cond': True},
              8: {'maskmem_features': y, 'obj_ptr': y, 'is_cond': False}}
    prepared = {0: native[0], 8: {'maskmem_features': torch.tensor([1., 2., 2.]), 'obj_ptr': None, 'is_cond': True},
                9: {'maskmem_features': y, 'obj_ptr': y, 'is_cond': False}}
    got = restoration.result(prepared, native, 10)
    points = got['restoration_frame_scores']
    assert [p['frame'] for p in points] == [0, 8, 9]
    assert [p['frames_before_switch'] for p in points] == [10, 2, 1]
    close(points[1]['maskmem_features']['r2'], .5)
    assert points[1]['obj_ptr']['status'] == 'missing_target_tensor'
    assert points[1]['native_is_cond'] is False and points[1]['target_is_cond'] is True
    assert points[2]['obj_ptr']['status'] == 'missing_native_frame'
    missing = restoration.frame_scores({0: native[0]}, native, 10)
    assert missing[1]['obj_ptr']['status'] == 'missing_target_frame'
    assert restoration.result({}, {}, 10)['restoration_status'] == 'empty_memory'
    assert not any(k.startswith('r2_') for k in got), '집계 결과는 아직 만들지 않는다'


def test_bank_summary():
    import json
    y1, y2 = torch.tensor([1., 2., 3.]), torch.tensor([10., 10., 10.])
    p1, p2 = torch.tensor([1., 2., 2.]), torch.tensor([11., 11., 11.])

    def entry(values):
        return {'maskmem_features': values, 'obj_ptr': values, 'is_cond': False}

    native = {1: entry(y1), 2: entry(y2), 0: entry(torch.tensor([100., 101., 102.]))}
    prepared = {1: entry(p1), 2: entry(p2)}
    raw = restoration.result(prepared, native, 3)
    before = json.dumps(raw, sort_keys=True)
    row = restoration.recompute([raw])[0]
    assert json.dumps(raw, sort_keys=True) == before
    expected = restoration.tensor_score(torch.cat([y1, y2]), torch.cat([p1, p2]))
    close(row['r2_maskmem_features'], expected['r2'])
    close(row['r2_obj_ptr'], expected['r2'])
    close(row['restoration_summary']['maskmem_features']['sst'], expected['sst'])
    assert abs(row['r2_maskmem_features'] - .5) > .1, '칸별 R² 단순 평균과 달라야 한다'
    stats = row['restoration_summary']['maskmem_features']
    assert stats['target_frame_count'] == stats['compared_frame_count'] == 2
    assert stats['native_only_frame_count'] == 1
    # 같은 충분통계량으로 반복 집계해도 값이 달라지지 않는다.
    assert restoration.recompute([row])[0] == row
    # 일부 Target 칸의 Native를 얻지 못한 경우 유리한 칸만 골라 대표값을 만들지 않는다.
    missing = restoration.result(prepared, {1: native[1]}, 3)
    scored = restoration.recompute([missing])[0]
    assert scored['r2_obj_ptr'] is None
    assert scored['restoration_summary']['obj_ptr']['status'] == 'incomplete_memory'
    # 칸 각각의 분산은 0이어도 전체 기억의 분산은 양수일 수 있다.
    varied = {1: entry(torch.ones(2)), 2: entry(torch.full((2,), 3.))}
    close(restoration.recompute([restoration.result(varied, varied, 3)])[0]['r2_obj_ptr'], 1)
    source = restoration.recompute([restoration.not_applicable(3)])[0]
    assert source['r2_obj_ptr'] is None
    assert source['restoration_summary']['obj_ptr']['status'] == 'source_only_no_target_memory'


def test_timing():
    clock = {'seconds': 0., 'gpu': 100., 'features_at': []}
    y = torch.tensor([1., 2., 3.])
    reference = {0: {'maskmem_features': y, 'obj_ptr': y, 'is_cond': True}}

    class Session:
        last = None
        closed = False

        def encode_prompts(self):
            clock['seconds'] += 2
            clock['gpu'] = 110

        def export_features(self):
            # 메모리 export를 오래 걸리게 만들어도 전환 비용에 포함되면 안 된다.
            clock['seconds'] += 100
            clock['features_at'].append(self.last)
            return reference

        def track(self, first, last):
            for frame in range(first, last + 1):
                self.last = frame
                clock['seconds'] += 3
                clock['gpu'] = 110 + frame
                yield SimpleNamespace(frame=frame, mask=np.zeros((2, 2), dtype=bool))

        def close(self):
            self.closed = True

    class Keeper:
        def keep(self, *args):
            pass

    original_result = restoration.result

    def slow_result(*args):
        clock['seconds'] += 1000
        return original_result(*args)

    for first, expected_seconds, expected_capture in ((9, 2, None), (7, 8, 8)):
        clock.update(seconds=0., gpu=100., features_at=[])
        session = Session()
        with patch.object(cost, 'now', lambda: clock['seconds']), \
             patch.object(cost, 'gpu_mb', lambda: 100.), \
             patch.object(cost, 'gpu_peak_mb', lambda: clock['gpu']), \
             patch.object(cost, 'reset_gpu_peak', lambda: None), \
             patch.object(restoration, 'result', slow_result):
            run = _run_base(SimpleNamespace(start=lambda video: session), None,
                            {'end': 10}, lambda session: first, 8, Keeper(),
                            restoration_reference=reference)
        assert clock['features_at'] == [expected_capture], clock
        assert run.restoration['restoration_measured_at'] == 8
        assert cost.cost_columns(run, 8)['switch_seconds'] == expected_seconds
        assert cost.cost_columns(run, 8)['switch_gpu_mb'] == (10 if first == 9 else 18)
        assert session.closed


if __name__ == '__main__':
    test_math()
    test_memory_frames()
    test_bank_summary()
    test_timing()
    print('OK: 프레임별 R², 음수/상수/누락, 전환 시점 측정과 비용 제외')
