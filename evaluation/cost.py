"""비용: 시간 · VRAM.

결과 줄에 남기는 비용 열:
  seconds_to_first_frame   [주] 첫 결과까지 시간: 전환 순간부터 전환 뒤 첫 프레임 결과를 낼 때까지 (준비 + 다시 보기 포함)
  seconds_per_frame_after  [추가] 전환 뒤 프레임당 평균 시간
  peak_vram_mb             [추가] 그 실행 동안 GPU 메모리 최고치 (켜 둔 두 모델 무게 포함)
모델이 전환 뒤 결과를 하나도 내지 않으면 (reset) 두 시간 열은 None.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import torch


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


class PeakVram:
    mb: float | None = None


@contextmanager
def peak_vram():
    box = PeakVram()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        yield box
    finally:
        if torch.cuda.is_available():
            box.mb = torch.cuda.max_memory_allocated() / 2 ** 20


def cost_columns(times: dict[int, float], setup_seconds: float, switch_frame: int,
                 peak_vram_mb, keeps_running: bool = False) -> dict:
    """times = {프레임: 그 프레임에 걸린 초}. keeps_running = 전환 없이 계속 도는 경우 (Source-only)."""
    first = switch_frame + 1
    after = [t for f, t in times.items() if f > switch_frame]
    if not after:
        to_first = per_frame = None
    else:
        to_first = times[first] if keeps_running else setup_seconds + sum(t for f, t in times.items() if f <= first)
        per_frame = sum(after) / len(after)
    return {
        "seconds_to_first_frame": to_first,
        "seconds_per_frame_after": per_frame,
        "peak_vram_mb": peak_vram_mb,
    }
