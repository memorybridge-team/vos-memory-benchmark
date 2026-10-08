"""GPU 없이 복사 독립성, 측정 구간, shard 및 불필요한 모델 로딩을 검증한다."""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import settings
import translator
from evaluation import cost, records, native
from evaluation import data
from evaluation.evaluate_video import evaluate_object
from evaluation.methods import METHODS
import fake_sam2
from model import sam2_runner
from test_end_to_end import setup, run_script, dataset_rows


def test_copy():
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        tensor = torch.arange(24, dtype=dtype).reshape(4, 6).T
        copy = sam2_runner._copy_to((tensor, None, [tensor]), 'cpu')
        for actual in (copy[0], copy[2][0]):
            assert torch.equal(actual, tensor) and actual.dtype == dtype
            assert actual.data_ptr() != tensor.data_ptr()
        copy[0].fill_(-1)
        assert torch.equal(copy[2][0], tensor) and bool((tensor >= 0).all())
        assert copy[1] is None and not copy[0].requires_grad
    grad = torch.ones(3, requires_grad=True)
    assert not sam2_runner._copy_to(grad, 'cpu').requires_grad


def test_measurement_limit():
    frames = [SimpleNamespace(frame=f) for f in range(10)]
    for limit in (0, 3, 10):
        ticks = []
        def now():
            ticks.append(1)
            return float(len(ticks))
        with patch.object(cost, 'now', now):
            outputs = list(cost.timed(frames, measure_frames=limit))
        assert [out.frame for out, _ in outputs] == list(range(10))
        assert [seconds for _, seconds in outputs] == [1.] * limit + [None] * (10 - limit)
        assert len(ticks) == 2 * limit


def test_selection_and_idle():
    with tempfile.TemporaryDirectory(prefix='vos_optimization_') as tmp:
        setup(Path(tmp))
        run_script('1_make_video_list.py')
        for dataset in data.DATASETS:
            all_videos = data.load_dataset(dataset)
            selected = data.load_dataset(dataset, names={all_videos[0].name})
            assert [v.name for v in selected] == [all_videos[0].name]
            assert selected[0].frame_paths == all_videos[0].frame_paths
            assert selected[0].mask_paths == all_videos[0].mask_paths
            assert data.load_dataset(dataset, names=set()) == []

        original = data.load_dataset
        requested = []
        def selected_load(dataset, *, names=None):
            requested.append(set(names))
            return original(dataset, names=names)
        with patch.object(data, 'load_dataset', selected_load):
            run_script('2_evaluate.py', '--dataset', 'vost_val', '--runs', '1', '--shard', '0/2')
            first = dataset_rows('vost_val')
            run_script('2_evaluate.py', '--dataset', 'vost_val', '--runs', '1', '--shard', '1/2')
        rows = dataset_rows('vost_val')
        assert requested == [{'101_cut_carrot'}, {'102_break_egg'}], requested
        assert {r['video'] for r in first} == requested[0]
        assert {r['video'] for r in rows} == requested[0] | requested[1]
        expected = sum(len(e['objects']) for e in data.load_video_list('vost_val')['videos']) * 2 * 6
        assert len(rows) == len(records.unique_rows(rows)) == expected
        assert all(r['runtime_revision'] == settings.EVALUATION_RUNTIME_REVISION for r in rows)

        def forbidden(*args, **kwargs):
            raise AssertionError('완료/빈 shard/Native 기준 재사용에 불필요한 모델 또는 데이터 로딩')
        with patch.object(data, 'load_dataset', forbidden), \
             patch.object(sam2_runner, 'load_runner', forbidden), \
             patch.object(translator, 'load', forbidden):
            run_script('2_evaluate.py', '--dataset', 'vost_val', '--runs', '1')
            run_script('2_evaluate.py', '--dataset', 'vost_val', '--runs', '1', '--shard', '3/4')

        # Native 행만 누락된 경우 이미 저장된 기준을 사용한다. Small/Base+/translator를 안 켠다.
        path = records.records_path('vost_val').with_name('vost_val.run1.shard0of2.jsonl')
        original_rows = records.read_rows(path)
        missing = [r for r in original_rows if r['baseline'] == 'full_replay' and r['object'] == 1]
        records.write_rows(path, [r for r in original_rows if r not in missing])
        with patch.object(sam2_runner, 'load_runner', forbidden), patch.object(translator, 'load', forbidden):
            run_script('2_evaluate.py', '--dataset', 'vost_val', '--runs', '1', '--shard', '0/2')
        restored = {records.row_key(r): r for r in records.read_rows(path)}
        assert all(restored[records.row_key(r)] == r for r in missing)
        assert len(dataset_rows('vost_val')) == expected

        # 메모리를 전달하지 않는 방법은 Small 기억을 꺼내지 않는다.
        entry = data.load_video_list('vost_val')['videos'][0]
        obj = entry['objects'][0]
        video = data.load_dataset('vost_val', names={entry['video']})[0]
        ref = native.load_references('vost_val')[native.case_key(video, obj, 1, settings.EVALUATION_SEED)]
        old_export = fake_sam2.FakeSession.export_memory
        def forbid_small_export(session):
            assert session.runner.name != 'small', '불필요한 Small memory export'
            return old_export(session)
        original_rows = {records.row_key(r): r for r in dataset_rows('vost_val')}
        for name in ('source_only', 'original_last_visible', 'original_replay_8'):
            method = next(m for m in METHODS if m.name == name)
            sw = next(sw for sw in obj['switches'] if sw['name'] == '75')
            with patch.object(fake_sam2.FakeSession, 'export_memory', forbid_small_export):
                got = evaluate_object(video, obj, fake_sam2.FakeRunner('small'),
                                      fake_sam2.FakeRunner('base_plus'), [method], native_reference=ref,
                                      pending_conditions={('75', sw['frame'], name)})
            assert len(got) == 1
            expected_row = original_rows[records.row_key(got[0])]
            for field in ('frame_scores', 'pre_switch_frame_scores', 'restoration_frame_scores',
                          'j', 'jf', 'failure_rate'):
                assert got[0][field] == expected_row[field], (name, field)



if __name__ == '__main__':
    test_copy()
    test_measurement_limit()
    test_selection_and_idle()
    print('OK: 독립된 단일 tensor 복사, 전환 측정 범위, shard 누락/중복, 완료/Native 재사용 로딩 생략')
