"""전환 비용: 준비(메모리 옮기기·변환·프롬프트 인코딩) + s까지의 replay.

Small에서 기억 꺼내기, 모델 로딩, 세션 생성과 s+1 이후 추적·채점은 제외한다.
GPU 메모리는 같은 전환 구간의 최고치에서 준비 직전 사용량을 뺀다.
Source-only는 전환이 없고 GPU가 없는 환경의 메모리 값은 None이다.
"""

from __future__ import annotations

import time

import torch


NO_COST = {"switch_seconds": None, "switch_gpu_mb": None}


def now() -> float:
    """GPU 작업이 다 끝난 뒤의 시각 (GPU는 비동기라 기다려야 정확함)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def timed(frames, measure_frames=None):
    """추적 결과를 한 장씩 받으면서 그 장을 만드는 데 걸린 시간을 같이 돌려준다.

    받은 뒤 채점하는 시간은 들어가지 않는다.
    measure_frames 이후에는 동기화/시간 측정을 생략하고 seconds=None을 반환한다.
    """
    it = iter(frames)
    index = 0
    while True:
        measure = measure_frames is None or index < measure_frames
        t0 = now() if measure else None
        try:
            item = next(it)
        except StopIteration:
            return
        seconds = now() - t0 if measure else None
        index += 1
        yield item, seconds


def gpu_mb() -> float | None:
    """지금 GPU 에 올라가 있는 양 (MB)."""
    return torch.cuda.memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None


def reset_gpu_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_mb() -> float | None:
    """마지막 reset_gpu_peak() 뒤 GPU 사용량 최고치 (MB)."""
    return torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None


def cost_columns(run, switch_frame: int) -> dict:
    """run = Base+ 실행 한 번. run.gpu_peaks[s] = 프레임 s 까지(준비 포함)의 최고치."""
    if switch_frame not in run.gpu_peaks:
        return dict(NO_COST)
    peak = run.gpu_peaks[switch_frame]
    return {
        "switch_seconds": run.setup_seconds + sum(seconds for f, seconds in run.times.items() if f <= switch_frame),
        "switch_gpu_mb": peak - run.gpu_before_mb if peak is not None and run.gpu_before_mb is not None else None,
    }
