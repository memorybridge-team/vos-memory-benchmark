"""DB 단독 재집계, 환경 독립 선택, 여러 실험/빈 결과 차단과 재개 격리."""

import contextlib
import csv
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import settings
from evaluation import store, records, native, data
from evaluation.scoring import recovery, restoration
from model import sam2_runner, sam2_check
import fake_data
from test_end_to_end import setup, run_script


def reject_cli(*args):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            run_script('3_make_tables.py', *args)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError('잘못된 실험/빈 선택을 허용했습니다.')


def table_files():
    root = Path(settings.OUTPUT_ROOT) / 'tables'
    return {p.name: p.read_bytes() for p in root.glob('*') if p.is_file()}


def server_setup(tmp):
    setup(tmp)
    with patch.object(fake_data, 'N', 33):
        fake_data.make_all(Path(settings.DATA_ROOT), settings.DATA_FOLDERS)
    checkpoints = tmp / 'checkpoints'
    checkpoints.mkdir()
    for spec in settings.MODELS.values():
        (checkpoints / spec['checkpoint']).write_bytes(b'fake-server-checkpoint')
    settings.SAM2_CHECKPOINT_DIR = str(checkpoints)
    report = Path(settings.OUTPUT_ROOT) / 'checks/sam2.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'runtime': {'gpu': 'server', 'benchmark_code': {'runner': 'before'}}}))
    run_script('1_make_video_list.py', '--datasets', 'pumavos')
    run_script('2_evaluate.py', '--dataset', 'pumavos', '--runs', '1')
    return store.experiment()[0]


def test_offline_portability():
    with tempfile.TemporaryDirectory(prefix='vos_offline_tables_') as tmp:
        server = Path(tmp) / 'server'
        server.mkdir()
        identity = server_setup(server)
        run_script('3_make_tables.py', '--runs', '1', '--skip-recovery-plots')
        expected = table_files()
        target = Path(tmp) / 'laptop/outputs'
        target.mkdir(parents=True)
        shutil.copyfile(store.database_path(), target / 'benchmark.sqlite')
        # 노트북에는 DB만 복사한다. Native .pt, 목록, RGB/GT, 체크포인트는 없다.
        with patch.object(settings, 'OUTPUT_ROOT', str(target)), \
                patch.object(settings, 'DATA_ROOT', '/notebook/other/data'), \
                patch.object(settings, 'SAM2_CHECKPOINT_DIR', '/notebook/missing/checkpoints'), \
                patch.object(settings, 'EVALUATION_REVISION', 999), \
                patch.object(settings, 'VIDEO_LIST_REVISION', 999), \
                patch.object(settings, 'SWITCH_FRACTIONS', (.5,)), \
                patch.object(settings, 'MIN_PRE_SWITCH_FRAMES', 999), \
                patch.object(settings, 'RECALL_J', .99), \
                patch.object(settings, 'J_MAIN_DATASETS', ()), \
                patch('evaluation.methods.METHODS', [SimpleNamespace(name=m, revision=999)
                                                    for m in store.select_experiment(identity)['baseline_revisions']]):
            # 실행/재개의 현재 환경 ID는 여전히 다르고, 기존 결과를 완료로 취급하지 않는다.
            assert store.experiment()[0] != identity
            assert records.done_keys('pumavos') == set()
            assert native.load_references('pumavos') == {}
            check = target / 'checks/sam2.json'; check.parent.mkdir()
            check.write_text('{broken-local-report')
            with patch.object(store, 'experiment', side_effect=AssertionError('집계에서 실행 환경을 재계산함')), \
                    patch.object(store, '_file_hash', side_effect=AssertionError('집계에서 체크포인트를 읽음')), \
                    patch.object(data, 'load_video_list', side_effect=AssertionError('로컬 목록을 읽음')), \
                    patch.object(native, 'load_memories', side_effect=AssertionError('기억 tensor를 읽음')), \
                    patch.object(sam2_runner, 'load_runner', side_effect=AssertionError('집계에서 모델을 켬')), \
                    patch.object(sam2_check, 'require_passed', side_effect=AssertionError('집계에서 preflight를 요구함')):
                run_script('3_make_tables.py', '--runs', '1', '--skip-recovery-plots')
                assert table_files() == expected
                run_script('3_make_tables.py', '--experiment-id', identity, '--runs', '1', '--skip-recovery-plots')
                assert table_files() == expected
                with store.connect() as conn:
                    assert conn.execute('SELECT DISTINCT experiment_id FROM analyses').fetchall()[0][0] == identity
                    assert conn.execute('SELECT COUNT(*) FROM experiments').fetchone()[0] == 1
                assert (target / 'analysis' / identity / 'recovery.seed0.runs1.median.ratio_of_means.jsonl').is_file()
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    run_script('3_make_tables.py', '--list-experiments')
                assert identity in output.getvalue() and 'seed=[0]' in output.getvalue()
            assert settings.EVALUATION_REVISION == settings.MIN_PRE_SWITCH_FRAMES == 999
            assert settings.RECALL_J == .99 and settings.J_MAIN_DATASETS == ()


def test_multiple_and_empty_selections():
    with tempfile.TemporaryDirectory(prefix='vos_experiment_choices_') as tmp:
        identity = server_setup(Path(tmp))
        run_script('3_make_tables.py', '--runs', '1', '--skip-recovery-plots')
        before = table_files()
        # 실제 두 번째 실험: 설정/평가 버전이 바뀐 실행은 별도 ID와 Native를 만든다.
        with patch.object(settings, 'EVALUATION_REVISION', settings.EVALUATION_REVISION + 1):
            run_script('2_evaluate.py', '--dataset', 'pumavos', '--runs', '1')
            second = store.experiment()[0]
        assert second != identity and len(store.list_experiments()) == 2
        reject_cli('--runs', '1')
        reject_cli('--experiment-id', 'missing-id', '--runs', '1')
        reject_cli('--experiment-id', identity, '--seed', '99', '--runs', '1')
        assert table_files() == before
        run_script('3_make_tables.py', '--experiment-id', identity, '--runs', '1', '--skip-recovery-plots')
        assert table_files() == before
        run_script('3_make_tables.py', '--experiment-id', second, '--runs', '1', '--skip-recovery-plots')
        assert json.loads((Path(settings.OUTPUT_ROOT) / 'tables/experiment.json').read_text())['experiment_id'] == second
        assert len(list((Path(settings.OUTPUT_ROOT) / 'analysis').glob('*/recovery.seed0.runs1.median.ratio_of_means.jsonl'))) == 2
        a = store.read_results(experiment_id=identity, visible_only=True)
        b = store.read_results(experiment_id=second, visible_only=True)
        for bad in (a[:1] + b[:1], a[:1]):
            try:
                store.save_analysis(bad, [1], 'median', 0, experiment_id=second)
            except ValueError:
                pass
            else:
                raise AssertionError('다른 실험으로 집계를 저장함')
        # 존재하는 실험이지만 결과가 없는 경우에도 이전 표는 보존한다.
        with store.connect() as conn:
            conn.execute('INSERT INTO experiments VALUES (?, ?)', ('0' * 64, store.dumps(store.select_experiment(identity)['configuration'])))
        second_tables = table_files()
        reject_cli('--experiment-id', '0' * 64, '--runs', '1')
        assert table_files() == second_tables


def test_missing_database():
    with tempfile.TemporaryDirectory(prefix='vos_no_database_') as tmp, patch.object(settings, 'OUTPUT_ROOT', tmp):
        tables = Path(tmp) / 'tables'; tables.mkdir()
        (tables / 'main.md').write_text('keep')
        reject_cli('--runs', '1')
        assert (tables / 'main.md').read_text() == 'keep'
        assert not store.database_path().exists()


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        test_offline_portability()
        test_multiple_and_empty_selections()
        test_missing_database()
    print('OK: DB만 복사한 재집계, 환경/버전 독립성, 실험 선택, 빈 선택의 표 보존, 재개 격리')
