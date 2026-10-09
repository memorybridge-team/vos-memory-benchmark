"""채점 동등성, CPU 큐 상한/오류/측정 격리와 SQLite 저장 완료를 검증한다."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import settings
import translator
from evaluation import cost, data, store
from evaluation.evaluate_video import Run, evaluate_object, _run_small, _run_base
from evaluation.methods import METHODS
from evaluation.scoring import jf
from evaluation.scoring.pipeline import FrameScorer, Keeper
import fake_sam2
from test_end_to_end import setup, run_script


def assert_no_workers():
    assert not any(t.name.startswith('vos-score') for t in threading.enumerate())


def test_exact_scores():
    rng = np.random.default_rng(91)
    n = 0
    for shape in ((1, 1), (1, 17), (19, 1), (32, 48), (127, 93)):
        for density in (0, .02, .4, 1):
            gt = rng.random(shape) < density
            for ignored in (None, np.zeros(shape, bool), rng.random(shape) < .3, np.ones(shape, bool)):
                prepared = jf.prepare_ground_truth(gt, ignored)
                assert gt.flags.writeable and (ignored is None or ignored.flags.writeable)
                for p_density in (0, .01, .5, 1):
                    pred = rng.random(shape) < p_density
                    expected = jf.FrameScore(jf.j_score(pred, gt, ignored), jf.f_score(pred, gt, ignored), bool(gt.any()))
                    assert jf.score_prepared(pred, prepared) == expected
                    n += 1
    return n


def test_cache():
    labels = np.zeros((40, 60), np.uint8); labels[5:20, 8:35] = 1
    calls = []
    def read(frame):
        calls.append(frame)
        return (labels.copy(), None) if frame != 9 else None
    video = SimpleNamespace(read_labels=read)
    size = jf.prepare_ground_truth(labels == 1).nbytes
    with patch.object(settings, 'GT_CACHE_MB', size / 1024 ** 2):
        scorer = FrameScorer(video, 1)
        assert scorer.score(0, labels == 1).jf == 1
        assert scorer.score(0, labels == 1).jf == 1
        assert calls == [0]
        scorer.score(1, labels == 1)
        scorer.score(0, labels == 1)
        assert calls == [0, 1, 0] and list(scorer.cache) == [0]
        assert scorer.cache_bytes <= scorer.cache_limit
        assert scorer.score(9, labels == 1) is None
    with patch.object(settings, 'GT_CACHE_MB', 0):
        scorer = FrameScorer(video, 1)
        scorer.score(0, labels == 1); scorer.score(0, labels == 1)
        assert scorer.cache_bytes == 0 and not scorer.cache


def test_queue_ownership_and_errors():
    labels = np.ones((10, 15), np.uint8)
    video = SimpleNamespace(read_labels=lambda frame: (labels, None))
    with patch.object(settings, 'SCORING_WORKERS', 2), patch.object(settings, 'SCORING_QUEUE_FRAMES', 2):
        scorer, run = FrameScorer(video, 1), Run()
        keeper = Keeper(scorer)
        entered, release = threading.Event(), threading.Event()
        score = scorer.score_timed
        def blocked(frame, pred):
            entered.set()
            assert release.wait(5), '마스크 소유권 검증 worker timeout'
            return score(frame, pred)
        with patch.object(scorer, 'score_timed', blocked), keeper.tracking(run):
            mask = np.ones(labels.shape, bool)
            keeper.keep(run, 0, mask)
            assert entered.wait(5)
            mask[:] = False
            release.set()
            for frame in range(1, 8):
                keeper.keep(run, frame, mask)
        assert run.scores[0].jf == 1 and all(run.scores[f].j == 0 for f in range(1, 8))
        assert run.cpu_profile['scoring']['peak_pending_frames'] <= 2
        assert run.cpu_profile['scoring']['tasks'] == 8 and keeper.executor is None
        assert_no_workers()
        # 동기/worker 오류 모두 점수0으로 바꾸지 않고 전파한다.
        for workers in (0, 2):
            with patch.object(settings, 'SCORING_WORKERS', workers):
                keeper = Keeper(scorer)
            with patch.object(scorer, 'score_timed', side_effect=ValueError('bad GT')):
                try:
                    with keeper.tracking(Run()):
                        keeper.keep(Run(), 0, mask)
                except ValueError as exc:
                    assert str(exc) == 'bad GT'
                else:
                    raise AssertionError('채점 오류를 무시함')
            assert_no_workers()
        # 추론 자체가 실패해도 이미 실행 중인 채점 작업은 종료한다.
        keeper = Keeper(scorer)
        try:
            with keeper.tracking(Run()):
                keeper.keep(Run(), 0, mask)
                raise RuntimeError('inference failed')
        except RuntimeError as exc:
            assert str(exc) == 'inference failed'
        assert_no_workers()


def fixture(tmp):
    setup(tmp)
    translator.load()
    run_script('1_make_video_list.py', '--datasets', 'vost_val')
    entry = data.load_video_list('vost_val')['videos'][0]
    return data.load_dataset('vost_val', names={entry['video']})[0], entry['objects'][0]


def test_measurement_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        video, obj = fixture(Path(tmp))
        keeper = Keeper(FrameScorer(video, obj['object']))
        prompt = video.object_mask(obj['start'], obj['object'])
        _run_small(fake_sam2.FakeRunner('small'), video, obj, prompt, keeper)
        assert_no_workers()
        observed = []
        score = keeper.scorer.score_timed
        def checked_score(frame, pred):
            observed.append((frame, threading.current_thread().name))
            return score(frame, pred)
        ticks = []
        def checked_now():
            assert_no_workers()
            ticks.append(1)
            return float(len(ticks))
        last = max(sw['frame'] for sw in obj['switches'])
        with patch.object(cost, 'now', checked_now), patch.object(keeper.scorer, 'score_timed', checked_score):
            run = _run_base(fake_sam2.FakeRunner('base_plus'), video, obj,
                            lambda session: (session.add_prompt(obj['start'], prompt) or obj['start']),
                            obj['start'] - 1, keeper, snapshot_frames={last})
        assert all(not name.startswith('vos-score') for f, name in observed if f <= last)
        assert any(name.startswith('vos-score') for f, name in observed if f > last)
        assert set(run.times) == set(range(obj['start'], last + 1))
        assert set(run.scores) == set(range(obj['start'], obj['end'] + 1))
        assert run.setup_seconds == 1 and all(t == 1 for t in run.times.values())
        assert_no_workers()


def test_end_to_end_and_save_guard():
    with tempfile.TemporaryDirectory() as tmp:
        video, obj = fixture(Path(tmp))
        rows = []
        for workers, cache in ((0, 0), (2, 128)):
            with patch.object(settings, 'SCORING_WORKERS', workers), patch.object(settings, 'GT_CACHE_MB', cache):
                result = evaluate_object(video, obj, fake_sam2.FakeRunner('small'),
                                         fake_sam2.FakeRunner('base_plus'), METHODS,
                                         save_native=store.save_native, save_result=lambda row: store.save_results([row]))
                rows.append(result)
                saved = store.read_results()
                assert len(saved) == len(result)
                for row in saved:
                    profile = row['cpu_profile']['scoring']
                    assert profile['tasks'] > 0 and profile['peak_pending_frames'] <= settings.SCORING_QUEUE_FRAMES
        fields = ('j', 'jf', 'n_frames', 'failure_rate', 'frame_scores', 'pre_switch_frame_scores',
                  'restoration_frame_scores', 'restoration_status')
        assert len(rows[0]) == len(rows[1])
        for a, b in zip(*rows):
            assert all(a.get(field) == b.get(field) for field in fields)
        source = next(r for r in rows[1] if r['baseline'] == 'source_only')
        assert source['cpu_profile']['scoring']['gt_cache_hits'] > 0
        # 첫 Native 추적의 비측정 프레임에서 worker가 실패하면 Native/결과 저장을 호출하지 않는다.
        saves = []
        score = FrameScorer.score_timed
        def broken(scorer, frame, mask):
            if frame == obj['end']:
                raise ValueError('GT decode error')
            return score(scorer, frame, mask)
        with patch.object(FrameScorer, 'score_timed', broken):
            try:
                evaluate_object(video, obj, fake_sam2.FakeRunner('small'), fake_sam2.FakeRunner('base_plus'),
                                METHODS, save_native=saves.append, save_result=saves.append)
            except ValueError as exc:
                assert str(exc) == 'GT decode error'
            else:
                raise AssertionError('실패한 채점 결과를 저장함')
        assert saves == []
        assert_no_workers()


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        count = test_exact_scores()
        test_cache()
        test_queue_ownership_and_errors()
        test_measurement_isolation()
        test_end_to_end_and_save_guard()
    print(f'OK: {count}개 J/F 정확 일치, 캐시/큐 상한, 마스크 소유권, worker 오류, 측정 격리, SQLite 저장 완료')
