"""CPU 채점 큐와 객체별 정답 LRU. CUDA/DB에는 접근하지 않는다."""

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock
from time import perf_counter
from uuid import uuid4

import numpy as np

import settings
from evaluation.scoring import jf


class FrameScorer:
    def __init__(self, video, obj_id):
        self.video, self.obj_id = video, obj_id
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.cache_limit = int(settings.GT_CACHE_MB * 1024 ** 2)
        if self.cache_limit < 0:
            raise ValueError('GT_CACHE_MB는 0 이상이어야 합니다.')
        self.lock = Lock()

    def _ground_truth(self, frame):
        # 같은 프레임의 중복 준비와 LRU 갱신을 막는다. 준비된 배열은 읽기 전용이다.
        with self.lock:
            if frame in self.cache:
                gt = self.cache.pop(frame)
                self.cache[frame] = gt
                return gt, {'gt_cache_hits': 1}
            t0 = perf_counter()
            raw = self.video.read_labels(frame)
            timings = {'gt_cache_misses': 1, 'gt_read_seconds': perf_counter() - t0}
            if raw is None:
                return None, timings
            labels, ignore = raw
            t0 = perf_counter()
            gt = jf.prepare_ground_truth(labels == self.obj_id, ignore)
            timings['gt_prepare_seconds'] = perf_counter() - t0
            if gt.nbytes <= self.cache_limit:
                while self.cache and self.cache_bytes + gt.nbytes > self.cache_limit:
                    _, old = self.cache.popitem(last=False)
                    self.cache_bytes -= old.nbytes
                self.cache[frame] = gt
                self.cache_bytes += gt.nbytes
            return gt, timings

    def score_timed(self, frame, pred):
        gt, timings = self._ground_truth(frame)
        if gt is None:
            return None, timings
        t0 = perf_counter()
        score = jf.score_prepared(pred, gt)
        timings['jf_seconds'] = perf_counter() - t0
        return score, timings

    def score(self, frame, pred):
        return self.score_timed(frame, pred)[0]


class Keeper:
    def __init__(self, scorer):
        self.scorer = scorer
        self.workers = settings.SCORING_WORKERS
        self.queue_limit = settings.SCORING_QUEUE_FRAMES
        if type(self.workers) is not int or self.workers < 0:
            raise ValueError('SCORING_WORKERS는 0 이상의 정수여야 합니다.')
        if type(self.queue_limit) is not int or self.queue_limit < 1:
            raise ValueError('SCORING_QUEUE_FRAMES는 1 이상의 정수여야 합니다.')
        self.pending = deque()
        self.executor = None
        self.active = False

    @contextmanager
    def tracking(self, run):
        """매 추적 종료 때 완료를 기다린다. 다음 측정 구간에는 worker가 없다."""
        if self.active:
            raise RuntimeError('동시에 두 추적을 채점할 수 없습니다.')
        self.active = True
        started = perf_counter()
        run.cpu_profile.update(scope='tracking_pass', tracking_id=uuid4().hex)
        self.stats = {'tasks': 0, 'peak_pending_frames': 0, 'wait_seconds': 0.0,
                      'gt_cache_hits': 0, 'gt_cache_misses': 0, 'gt_read_seconds': 0.0,
                      'gt_prepare_seconds': 0.0, 'jf_seconds': 0.0,
                      'workers': self.workers, 'queue_limit_frames': self.queue_limit,
                      'cache_limit_bytes': self.scorer.cache_limit}
        try:
            yield
            self.flush()
        finally:
            # 결과/추론 오류가 있어도 모든 실행 중 작업이 끝난 뒤 빠져나간다.
            for _, _, future in self.pending:
                future.cancel()
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.pending.clear()
            self.active = False
            run.cpu_profile['scoring'] = dict(self.stats)
            run.cpu_profile['scoring']['cache_retained_bytes'] = self.scorer.cache_bytes
            run.cpu_profile['wall_seconds'] = perf_counter() - started

    def _store(self, run, frame, result):
        score, timings = result
        if score is not None:
            run.scores[frame] = score
        for key, value in timings.items():
            self.stats[key] += value
        self.stats['tasks'] += 1

    def _collect(self):
        run, frame, future = self.pending.popleft()
        t0 = perf_counter()
        try:
            result = future.result()  # worker 오류는 저장 전에 호출자에게 전파한다.
        finally:
            self.stats['wait_seconds'] += perf_counter() - t0
        self._store(run, frame, result)

    def flush(self):
        while self.pending:
            self._collect()

    def keep(self, run, frame, mask, *, parallel=True):
        if not self.active:
            raise RuntimeError('tracking 범위 밖에서 채점을 요청했습니다.')
        if not parallel or self.workers == 0:
            self.flush()
            self._store(run, frame, self.scorer.score_timed(frame, mask))
            return
        if len(self.pending) >= self.queue_limit:
            self._collect()
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix='vos-score')
        # SAM2/테스트 모델이 다음 프레임에서 원본 마스크를 재사용해도 점수가 바뀌지 않는다.
        owned_mask = np.array(mask, dtype=bool, copy=True)
        future = self.executor.submit(self.scorer.score_timed, frame, owned_mask)
        self.pending.append((run, frame, future))
        self.stats['peak_pending_frames'] = max(self.stats['peak_pending_frames'], len(self.pending))
