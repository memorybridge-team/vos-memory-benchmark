"""전체 프레임 원점수, 트랜잭션, 재개, 집계 정의 분리와 실제 25% 실행."""

import copy
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import runpy
import sqlite3
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import settings
import translator
from evaluation import data, native, records, store
from evaluation.evaluate_video import evaluate_object
from evaluation.methods import METHODS
from evaluation.scoring import main_metrics, recovery, restoration
from evaluation.switches import eligible_objects, switch_points
from model import sam2_runner, sam2_check
import fake_data
import fake_sam2
from test_end_to_end import setup, run_script


def fails(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError('실패해야 하는 작업이 통과했습니다.')


def test_eligibility_before_loading():
    assert switch_points(0, 12) == []
    assert switch_points(0, 10) == []
    assert switch_points(0, 9) == []
    assert switch_points(4, 36)[0] == {'name': '25', 'frame': 12}
    bad = {'object': 1, 'start': 0, 'end': 12, 'switches': [{'name': '50', 'frame': 6}]}
    assert eligible_objects([{'video': 'v', 'objects': [bad]}]) == []
    with tempfile.TemporaryDirectory() as tmp, patch.object(settings, 'OUTPUT_ROOT', tmp), \
            patch.object(data, 'load_video_list', return_value={'videos': [{'video': 'v', 'objects': [bad]}]}), \
            patch.object(data, 'load_dataset', side_effect=AssertionError('불필요한 데이터 로딩')), \
            patch.object(sam2_runner, 'load_runner', side_effect=AssertionError('불필요한 모델 로딩')), \
            patch.object(sam2_check, 'require_passed', side_effect=AssertionError('불필요한 검사')):
        run_script('2_evaluate.py', '--dataset', 'vost_val', '--runs', '1')
        assert not store.database_path().exists()
    row = dict(dataset='vost_val', video='v', object=1, start=0, end=12, run_id=1, seed=0,
               baseline='source_only', baseline_revision=1, switch_name='50', switch_frame=6,
               evaluation_revision=settings.EVALUATION_REVISION, native_reference_id='id')
    assert records.current_rows([row]) == []
    # 75% 자체는 9프레임이어도 25%가 짧으므로 모든 방법에서 제외한다.
    rows = [dict(row, baseline=m.name, baseline_revision=m.revision,
                 switch_name='75', switch_frame=9) for m in METHODS]
    assert records.current_rows(rows) == []
    assert evaluate_object(SimpleNamespace(name='v'), bad, None, None, METHODS) == []
    exact = dict(bad, end=32, switches=switch_points(0, 32))
    assert len(eligible_objects([{'video': 'v', 'objects': [exact]}])[0]['objects'][0]['switches']) == 3
    partial = dict(exact, switches=exact['switches'][1:])
    assert eligible_objects([{'video': 'v', 'objects': [partial]}]) == []
    # 이전 실험은 저장 당시의 전환별 규칙으로 별도 집계한다.
    legacy = dict(evaluation_revision=6, switch_fractions=[.25,.5,.75], min_pre_switch_frames=8)
    assert len(records.current_rows([dict(r, evaluation_revision=6) for r in rows], configuration=legacy)) == len(METHODS)


def test_all_frames_and_atomic_resume():
    with tempfile.TemporaryDirectory(prefix='vos_sqlite_') as tmp:
        setup(Path(tmp))
        run_script('1_make_video_list.py')
        video = data.load_dataset('vost_val')[0]
        # 희소 GT 전용 시험. 실제 데이터 로더의 정합성 검사는 유지한다.
        video.mask_paths.pop(20)
        obj = dict(object=2, start=0, end=video.num_frames - 1,
                   switches=switch_points(0, video.num_frames - 1), extra_labels=['OCC'])
        translator.load()
        refs = []
        def save(ref):
            store.save_native(ref)
            refs.append(ref)
        rows = evaluate_object(video, obj, fake_sam2.FakeRunner('small'), fake_sam2.FakeRunner('base_plus'),
                               METHODS, save_native=save, save_result=lambda r: store.save_results([r]))
        assert len(rows) == 18 and len(refs) == 1
        raw = store.read_results('vost_val')
        visible = store.read_results('vost_val', visible_only=True)
        for row in raw:
            points = row['pre_switch_frame_scores'] + row['frame_scores']
            assert [p['frame'] for p in points] == list(range(video.num_frames))
            missing = next(p for p in points if p['frame'] == 20)
            assert missing['has_gt'] == 0 and missing['gt_visible'] is None
            assert missing['j'] is missing['f'] is missing['jf'] is None
            hidden = next(p for p in points if p['frame'] == 15)
            assert hidden['has_gt'] == 1 and hidden['gt_visible'] == 0 and hidden['j'] is not None
            assert len(row['frame_times']) == video.num_frames
            assert next(p for p in row['frame_times'] if p['frame'] == 29)['seconds'] is None
            partner = next(r for r in visible if r['result_id'] == row['result_id'])
            assert all(p['has_gt'] and p['gt_visible'] for p in partner['frame_scores'] + partner['pre_switch_frame_scores'])
            assert row['n_frames'] == len(partner['frame_scores'])
            assert row['j'] == main_metrics.mean(p['j'] for p in partner['frame_scores'])
            assert row['jf'] == main_metrics.mean(p['jf'] for p in partner['frame_scores'])
        ref = next(iter(native.load_references('vost_val').values()))
        assert len(ref['scores']) == video.num_frames and len(ref['frame_times']) == video.num_frames
        assert native.scores(ref).keys() == set(range(video.num_frames)) - {20}
        # 자식 행 저장 중 실패하면 해당 결과의 부모와 자식 모두 롤백한다.
        broken = copy.deepcopy(rows[0]); broken['run_id'] = 2
        bad_ref = copy.deepcopy(ref); bad_ref['run_id'] = 2
        bad_ref['native_reference_id'] = 'a' * 32
        native.save_memories(bad_ref, native.load_memories(ref)); store.save_native(bad_ref)
        broken['native_reference_id'] = bad_ref['native_reference_id']
        broken['frame_scores'][0]['phase'] = 'unused'
        broken['frame_scores'][0]['has_gt'] = 2
        fails(lambda: store.save_results([broken]), sqlite3.IntegrityError)
        assert len(store.read_results('vost_val')) == 18
        assert len(records.done_keys('vost_val', run_ids=[1])) == 18
        fails(lambda: store.save_results([rows[0]]), sqlite3.IntegrityError)
        assert len(store.read_results('vost_val')) == 18
        with store.connect() as conn:
            assert conn.execute('PRAGMA foreign_keys').fetchone()[0] == 1
            assert not conn.execute('PRAGMA foreign_key_check').fetchall()
            assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        before = store.read_results('vost_val')
        run_one = restoration.recompute(recovery.recompute(visible, run_ids=[1]))
        a = store.save_analysis(run_one, [1], 'median', 0)
        pending = restoration.recompute(recovery.recompute(visible, run_ids=[1, 2, 3]))
        b = store.save_analysis(pending, [1, 2, 3], 'median', 0)
        assert a != b and store.read_results('vost_val') == before
        with store.connect() as conn:
            assert conn.execute('SELECT COUNT(*) FROM analyses').fetchone()[0] == 2
            assert conn.execute('SELECT COUNT(*) FROM recovery_summaries').fetchone()[0] == 36
            assert conn.execute('SELECT post_recovery_j FROM recovery_summaries WHERE analysis_id=? LIMIT 1', (b,)).fetchone()[0] is None
        # 같은 실험에서 방법 revision만 바뀌어도 이전 행을 덮어쓰지 않는다.
        revised = copy.deepcopy(rows[0]); revised['baseline_revision'] += 1
        store.save_results([revised])
        with patch('evaluation.methods.METHODS', [SimpleNamespace(name=revised['baseline'], revision=revised['baseline_revision'])]):
            selected = store.read_results('vost_val')
            assert len(selected) == 1 and selected[0]['baseline_revision'] == revised['baseline_revision']
        assert len(store.read_results('vost_val')) == 18
        # 새 revision은 같은 DB에 보관하고 현재 조회에서는 이전 결과를 제외한다.
        with patch.object(settings, 'EVALUATION_REVISION', settings.EVALUATION_REVISION + 1):
            assert store.read_results('vost_val') == [] and records.done_keys('vost_val') == set()
            changed_ref = copy.deepcopy(ref)
            changed_ref['native_reference_id'] = 'b' * 32
            changed_ref['evaluation_revision'] = settings.EVALUATION_REVISION
            native.save_memories(changed_ref, native.load_memories(ref)); store.save_native(changed_ref)
            changed_row = copy.deepcopy(rows[0]); changed_row['native_reference_id'] = changed_ref['native_reference_id']
            changed_row['evaluation_revision'] = settings.EVALUATION_REVISION
            store.save_results([changed_row])
            assert len(store.read_results('vost_val')) == 1
        assert len(store.read_results('vost_val')) == 18
        native.memory_path(ref).unlink()
        fails(lambda: native.load_references('vost_val'))


def test_concurrent_initialization():
    # 두 shard가 빈 DB를 동시에 열어도 완성된 스키마만 노출한다.
    with tempfile.TemporaryDirectory(prefix='vos_db_writers_') as tmp, \
            patch.object(settings, 'OUTPUT_ROOT', tmp):
        barrier = threading.Barrier(2)
        def writer(index):
            barrier.wait()
            with store.connect() as conn:
                conn.execute('INSERT INTO experiments VALUES (?, ?)', (str(index), '{}'))
                assert conn.execute('PRAGMA foreign_keys').fetchone()[0] == 1
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(writer, (1, 2)))
        with store.connect() as conn:
            assert conn.execute('SELECT COUNT(*) FROM experiments').fetchone()[0] == 2
            assert conn.execute('PRAGMA user_version').fetchone()[0] == store.SCHEMA_VERSION
            assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'


def test_prediction_gaps_are_errors():
    with tempfile.TemporaryDirectory(prefix='vos_prediction_gap_') as tmp:
        setup(Path(tmp))
        video = data.load_dataset('vost_val')[0]
        obj = dict(object=1, start=0, end=video.num_frames - 1, switches=switch_points(0, video.num_frames - 1))
        original = fake_sam2.FakeSession.track
        def skip(self, first, last):
            for out in original(self, first, last):
                if self.runner.name != 'small' or out.frame != 2:
                    yield out
        with patch.object(fake_sam2.FakeSession, 'track', skip):
            fails(lambda: evaluate_object(video, obj, fake_sam2.FakeRunner('small'),
                  fake_sam2.FakeRunner('base_plus'), METHODS[:1], save_native=store.save_native,
                  save_result=lambda r: store.save_results([r])))
        # 모델 출력 누락을 GT 누락 NULL 행으로 바꾸지 않는다. Native는 재개에 남긴다.
        assert store.read_results('vost_val') == []
        assert len(native.load_references('vost_val')) == 1


def test_25_and_repeat_resume():
    with tempfile.TemporaryDirectory(prefix='vos_switch25_') as tmp:
        # 33프레임이면 최초 등장 0인 객체는 25%에서 정확히 8프레임을 본다.
        with patch.object(fake_data, 'N', 33):
            setup(Path(tmp))
        run_script('1_make_video_list.py', '--datasets', 'pumavos')
        assert data.load_video_list('pumavos')['videos'][0]['objects'][0]['switches'][0] == {'name': '25', 'frame': 8}
        for runs in (1, 2, 3):
            run_script('2_evaluate.py', '--dataset', 'pumavos', '--runs', str(runs))
            raw = store.read_results('pumavos')
            assert len(raw) == runs * 6 * len(METHODS)
            assert len(native.load_references('pumavos')) == runs * 2
        before = store.read_results('pumavos')
        with patch.object(sam2_runner, 'load_runner', side_effect=AssertionError('완료 결과 재실행')):
            run_script('2_evaluate.py', '--dataset', 'pumavos', '--runs', '3')
        assert store.read_results('pumavos') == before
        run_script('3_make_tables.py', '--runs', '3', '--skip-recovery-plots')
        with (Path(settings.OUTPUT_ROOT) / 'tables/summary.csv').open(encoding='utf-8-sig') as stream:
            summary = list(csv.DictReader(stream))
        assert {p['switch_name'] for p in summary} == {'25', '50', '75'}
        assert all(int(r['j_run_count']) == 3 for r in summary)
        assert store.read_results('pumavos') == before


if __name__ == '__main__':
    test_eligibility_before_loading()
    test_all_frames_and_atomic_resume()
    test_concurrent_initialization()
    test_prediction_gaps_are_errors()
    test_25_and_repeat_resume()
    print('OK: 객체 전체 사전 제외, 전체 프레임/NULL, 원자적 SQLite 재개, 버전/집계 정의 분리, 25%, 1→2→3회')
