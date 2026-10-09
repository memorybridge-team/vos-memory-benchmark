"""CPU 채점 스케줄 변경이 회차별 원점수·RNG·집계 표본을 바꾸지 않는지 검증한다.

실제 CUDA 추론 검증을 대체하지 않는다. 기준 채점은 캐시 없는 기존 J/F 함수다.
"""

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import settings
import translator
from evaluation import evaluate_video, store
from evaluation.data.common import Video
from evaluation.methods import METHODS
from evaluation.scoring import jf, recovery, restoration
from evaluation.scoring.pipeline import FrameScorer
from evaluation.tables import summaries, temporal, recovery_curves
from model.sam2_runner import FrameOut
import fake_data
import fake_sam2


class ReferenceScorer(FrameScorer):
    def score_timed(self, frame, pred):
        raw = self.video.read_labels(frame)
        if raw is None:
            return None, {'gt_cache_misses': 1}
        labels, ignore = raw
        gt = labels == self.obj_id
        return jf.FrameScore(jf.j_score(pred, gt, ignore), jf.f_score(pred, gt, ignore),
                             bool(gt.any())), {'gt_cache_misses': 1}


class StochasticSession(fake_sam2.FakeSession):
    def track(self, first, last):
        for out in super().track(first, last):
            # 세 RNG를 사용하고 일부 예측을 실제 0점으로 만든다. 오류로 제외하지 않는다.
            shift = random.randrange(3) - 1
            fail = np.random.random() < .15
            shift_y = int(torch.randint(-1, 2, ()).item())
            if out.frame not in self.cond:
                mask = np.zeros_like(out.mask) if fail else np.roll(out.mask, (shift_y, shift), (0, 1))
                color = fake_sam2._color_of(self.non_cond[out.frame])
                self.non_cond[out.frame] = fake_sam2._entry(color, mask)
                out = FrameOut(out.frame, mask, bool(mask.any()))
            yield out


class StochasticRunner(fake_sam2.FakeRunner):
    def start(self, video):
        return StochasticSession(self, video)


def fixture(root):
    videos = []
    for name in ('case_a', 'case_b'):
        folder = root / name
        folder.mkdir()
        frames, masks = [], {}
        for frame in range(65):
            image, labels = fake_data.draw(frame % 30, True)
            path = folder / f'{frame:05d}.png'
            Image.fromarray(image).save(path)
            frames.append(path)
            if frame not in (17, 35):
                gt = folder / f'{frame:05d}_gt.png'
                Image.fromarray(labels).save(gt)
                masks[frame] = gt
        videos.append(Video('vost_val', name, frames, masks, 255))
    objects = [{'object': obj, 'start': 0, 'end': 64,
                'extra_labels': [], 'switches': [{'name': name, 'frame': f}
                    for name, f in (('25', 16), ('50', 32), ('75', 48))]}
               for obj in (1, 2)]
    return videos, objects


def semantic(rows):
    """시간 숫자는 같다고 가정하지 않고, 측정 대상 프레임의 유무만 비교한다."""
    result = []
    for row in rows:
        item = {k: v for k, v in row.items() if k not in
                ('cpu_profile', 'pre_cpu_profile', 'restoration_cache_profile',
                 'result_id', 'experiment_id', 'native_reference_id',
                 'switch_seconds', 'switch_gpu_mb', 'frame_times')}
        item['switch_time_measured'] = row['switch_seconds'] is not None
        item['frame_time_measured'] = [(p['frame'], p['seconds'] is not None,
                                      p['gpu_peak_mb'] is not None) for p in row['frame_times']]
        result.append(item)
    return result


def rng_state():
    state = np.random.get_state()
    return (random.getstate(), (state[0], state[1].tobytes(), *state[2:]),
            torch.get_rng_state().numpy().tobytes())


def quality_tables(rows, statistic):
    analysed = restoration.recompute(recovery.recompute(store.visible_rows(rows), (1, 2, 3), statistic))
    # 시간 숫자는 동등성 판정 대상이 아니다.
    analysed = [dict(r, switch_seconds=None, switch_gpu_mb=None) for r in analysed]
    return (summaries.build(analysed), temporal.build(analysed),
            recovery_curves.build(analysed, run_ids=(1, 2, 3)),
            recovery.reference_rows(analysed, (1, 2, 3), statistic))


def test_repeated_cohort():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        videos, objects = fixture(root)
        reference = reference_rng = None
        comparisons = 0
        for mode, workers, queue, cache in (
                ('reference', 0, 4, 0), ('default', 1, 4, 128),
                ('eviction', 4, 4, .02), ('no_cache', 4, 1, 0)):
            rows, states = [], []
            with ExitStack() as stack:
                for key, value in dict(SCORING_WORKERS=workers, SCORING_QUEUE_FRAMES=queue,
                                       GT_CACHE_MB=cache, EVALUATION_RUNS=3,
                                       RGB_CACHE_MB=0 if mode == 'reference' else 256,
                                       RESTORATION_CACHE_MB=0 if mode == 'reference' else 128,
                                       BENCHMARK_SKIP_VISIBLE=mode != 'reference').items():
                    stack.enter_context(patch.object(settings, key, value))
                stack.enter_context(patch.object(translator, '_translator', fake_sam2.FakeTranslator()))
                if mode == 'reference':
                    stack.enter_context(patch.object(evaluate_video, 'FrameScorer', ReferenceScorer))
                for run_id in (1, 2, 3):
                    for video in videos:
                        for obj in objects:
                            rows.extend(evaluate_video.evaluate_object(video, obj,
                                StochasticRunner('small'), StochasticRunner('base_plus'),
                                METHODS, run_id=run_id))
                            states.append(rng_state())
            expected_count = 3 * 2 * 2 * 3 * len(METHODS)
            assert len(rows) == expected_count
            for row in rows:
                points = row['pre_switch_frame_scores'] + row['frame_scores']
                assert len(points) == 65
                assert [p['frame'] for p in points if not p['has_gt']] == [17, 35]
                assert all(p['j'] is None and p['f'] is None for p in points if not p['has_gt'])
            assert any(p['j'] == 0 and p['gt_visible'] for r in rows for p in r['frame_scores'])
            assert any(not p['gt_visible'] and p['has_gt'] for r in rows for p in r['frame_scores'])
            native = [r for r in rows if r['baseline'] == 'full_replay' and r['switch_name'] == '25']
            assert len({json.dumps(r['frame_scores'], sort_keys=True) for r in native}) > 3
            current = semantic(rows)
            if reference is None:
                reference, reference_rng = rows, states
            else:
                assert current == semantic(reference), mode
                assert states == reference_rng, f'{mode}: 추론 RNG 상태 변경'
                for statistic in ('median', 'mean'):
                    assert quality_tables(rows, statistic) == quality_tables(reference, statistic), (mode, statistic)
                comparisons += 1
        return {'rows_per_mode': expected_count, 'frames_per_row': 65,
                'comparison_modes': comparisons, 'exact_raw_score_records': comparisons * expected_count * 65,
                'switches': [25, 50, 75], 'runs': 3, 'videos': 2, 'objects_per_video': 2,
                'rng_exact': True, 'mean_and_median_tables_exact': True,
                'missing_gt_zero_scores_visibility_preserved': True}


def test_high_resolution_concurrency():
    rng = np.random.default_rng(826)
    compared = 0
    for shape in ((480, 720), (1080, 1920)):
        gt = np.zeros(shape, bool)
        gt[20:shape[0] - 20, 30:shape[1] // 2] = True
        ignore = rng.random(shape) < .002
        predictions = [np.roll(gt, (dy, dx), (0, 1)) for dy, dx in ((0, 0), (2, 3), (6, 8), (12, 16))]
        prepared = jf.prepare_ground_truth(gt, ignore)
        expected = [jf.FrameScore(jf.j_score(p, gt, ignore), jf.f_score(p, gt, ignore), True)
                    for p in predictions]
        # 같은 읽기 전용 정답에 동시 접근하며 OpenCV 거리 변환도 겹치게 한다.
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(jf.score_prepared, p, prepared) for _ in range(8) for p in predictions]
            assert [job.result() for job in jobs] == expected * 8
        assert not prepared.mask.flags.writeable and not prepared.distance.flags.writeable
        compared += len(jobs)
    return compared


if __name__ == '__main__':
    report = test_repeated_cohort()
    report['high_resolution_concurrent_exact_pairs'] = test_high_resolution_concurrency()
    print(json.dumps(report, ensure_ascii=False, indent=2))
