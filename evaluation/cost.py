"""비용: 시간 · GPU 메모리. Base+ 실행 한 번(evaluate_video.Run)으로 결과 줄의 비용 열을 만든다.

결과 줄에 남기는 비용 열:
  seconds              [주] 시간: Base+ 가 한 일 전부 = 준비(프롬프트·기억 넣기) + 추적한 모든 프레임
                       Full Replay = Base+ 로 처음 ~ 끝 전체 / 다른 비교군 = 넘기기(Replay-K 는 다시 보기 포함) + s+1 ~ 끝.
                       Small 에서 기억 꺼내기(export)는 넣지 않는다.
  extra_switch_gpu_mb  [추가] 전환 GPU 메모리(MB): 전환 구간 GPU 메모리 최고치 − 전환 직전 사용량.
                       전환 구간 = Small 이 s 를 끝낸 뒤 ~ Base+ 가 s+1 처리 준비를 마칠 때 (바꿔 넣기 + 다시 보기).
                       전환 직전 = Base+ 세션을 연 직후.
Source-only(전환 없음)와 reset(전환 뒤 아무것도 추적 안 함)은 둘 다 None. GPU 가 없으면 GPU 열은 None.
"""

from __future__ import annotations

import time

import torch

NO_COST = {"seconds": None, "extra_switch_gpu_mb": None}


def now() -> float:
    """GPU 작업이 다 끝난 뒤의 시각 (GPU는 비동기라 기다려야 정확함)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def timed(frames):
    """추적 결과를 한 장씩 받으면서 그 장을 만드는 데 걸린 시간을 같이 돌려준다.

    받은 뒤 채점하는 시간은 들어가지 않는다.
    """
    it = iter(frames)
    while True:
        t0 = now()
        try:
            item = next(it)
        except StopIteration:
            return
        yield item, now() - t0


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
    if not run.times:
        return dict(NO_COST)
    peak = run.gpu_peaks[switch_frame]
    return {
        "seconds": run.setup_seconds + sum(run.times.values()),
        "extra_switch_gpu_mb": None if peak is None else peak - run.gpu_before_mb,
    }
