"""전환 시점: Small이 어느 프레임까지 보고 Base+ 로 넘기는가.

전환 프레임 s = Small이 마지막으로 본 프레임. Base+ 는 s+1 부터 이어받는다.

    A (주)   객체 구간 [처음 보인 프레임, 영상 끝]의 25 / 50 / 75% 지점
    B (추가) 객체가 한동안 안 보이다가 다시 나타나기 바로 전 프레임
             → 정답이 모든 프레임에 있어야 알 수 있음 (MOSEv2 valid 는 불가)
"""

from __future__ import annotations

import settings


def switch_points_a(start: int, end: int) -> list[dict]:
    points = []
    for frac in settings.SWITCH_A_FRACTIONS:
        frame = start + round(frac * (end - start))
        frame = min(max(frame, start + 1), end - 1)
        points.append({"set": "A", "name": f"A{round(frac * 100)}", "frame": frame})
    return points


def switch_points_b(visible: dict[int, bool], start: int, end: int) -> list[dict]:
    """visible = {정답 프레임: 객체가 보이는가}. 안 보인 기간이 긴 것부터 고른다."""
    events = []          # (연속으로 안 보인 정답 프레임 수, 다시 나타난 프레임)
    absent = 0
    for frame in sorted(f for f in visible if f > start):
        if not visible[frame]:
            absent += 1
            continue
        if absent >= settings.SWITCH_B_MIN_ABSENT:
            events.append((absent, frame))
        absent = 0
    events.sort(key=lambda e: (-e[0], e[1]))

    points = []
    for absent_frames, reappear in events[:settings.SWITCH_B_PER_OBJECT]:
        frame = reappear - 1
        if start < frame < end:
            points.append({"set": "B", "name": f"B{len(points) + 1}", "frame": frame,
                           "absent_gt_frames": absent_frames})
    return points
