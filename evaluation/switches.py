"""전환 시점: Small이 어느 프레임까지 보고 Base+ 로 넘기는가.

전환 프레임 s = Small이 마지막으로 본 프레임. Base+ 는 s+1 부터 이어받는다.
객체 구간 [처음 보인 프레임, 영상 끝]의 25 / 50 / 75% 지점.
하나라도 s-start가 8프레임 미만이면 객체의 모든 전환을 제외한다. 정확히 8은 포함한다.
"""

from __future__ import annotations

import settings


def switch_name(fraction: float) -> str:
    return str(round(fraction * 100))


def switch_points(start: int, end: int, *, fractions=None, min_pre_frames=None, whole_object=True) -> list[dict]:
    points = []
    for frac in settings.SWITCH_FRACTIONS if fractions is None else fractions:
        frame = start + round(frac * (end - start))
        frame = min(max(frame, start + 1), end - 1)
        if not eligible_switch(start, end, frame, min_pre_frames=min_pre_frames):
            if whole_object:
                return []
            continue
        points.append({"name": switch_name(frac), "frame": frame})
    return points


def eligible_switch(start: int, end: int, frame: int, *, min_pre_frames=None) -> bool:
    minimum = settings.MIN_PRE_SWITCH_FRAMES if min_pre_frames is None else min_pre_frames
    return start <= frame < end and frame - start >= minimum


def eligible_objects(entries):
    """고정 목록을 실행 직전에 다시 검사한다. 입력 목록은 변경하지 않는다."""
    out = []
    for entry in entries:
        objects = []
        for obj in entry['objects']:
            switches = switch_points(obj['start'], obj['end'])
            if switches and obj['switches'] == switches:
                objects.append(obj)
        if objects:
            out.append({**entry, 'objects': objects})
    return out
