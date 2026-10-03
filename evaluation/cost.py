"""비용: 시간 · VRAM.

결과 줄에 남기는 비용 열:
  switch_seconds           [주] 전환 지연: Small이 s 까지 처리한 직후부터 Base+ 가 s+1 을 처리할 준비가 끝날 때까지
                           = 옮기기 + Base+ 형태로 바꿔 넣기 (setup_seconds) + 다시 보기 (s 이하 프레임 시간).
                           Small 메모리 꺼내기(export)와 s+1 처리는 뺀다 (SCD 식 T_transfer 와 같은 범위).
  seconds_per_frame_after  [추가] 전환 뒤 프레임당 평균 시간
  peak_vram_mb             [추가] 그 실행 동안 GPU 메모리 최고치 (켜 둔 두 모델 무게 포함)
전환이 없거나 (Source-only) 전환 뒤 결과를 하나도 내지 않으면 (reset) switch_seconds 는 None.
reset 은 seconds_per_frame_after 도 None.
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
    after = [t for f, t in times.items() if f > switch_frame]
    if not after:
        switch = per_frame = None
    else:
        replay = sum(t for f, t in times.items() if f <= switch_frame)
        switch = None if keeps_running else setup_seconds + replay
        per_frame = sum(after) / len(after)
    return {
        "switch_seconds": switch,
        "seconds_per_frame_after": per_frame,
        "peak_vram_mb": peak_vram_mb,
    }
