"""전환 시점: Small이 어느 프레임까지 보고 Base+ 로 넘기는가.

전환 프레임 s = Small이 마지막으로 본 프레임. Base+ 는 s+1 부터 이어받는다.
객체 구간 [처음 보인 프레임, 영상 끝]의 25 / 50 / 75% 지점 (settings.SWITCH_FRACTIONS).
전환 이름은 비율 숫자 ("25", "50", "75") — 결과 줄과 MOSEv2 제출 파일 이름에 쓴다.
"""

from __future__ import annotations

import settings


def switch_name(fraction: float) -> str:
    return str(round(fraction * 100))


def switch_points(start: int, end: int) -> list[dict]:
    points = []
    for frac in settings.SWITCH_FRACTIONS:
        frame = start + round(frac * (end - start))
        frame = min(max(frame, start + 1), end - 1)
        points.append({"name": switch_name(frac), "frame": frame})
    return points
