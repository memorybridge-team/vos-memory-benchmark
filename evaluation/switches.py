"""전환 시점: Small이 어느 프레임까지 보고 Base+ 로 넘기는가.

전환 프레임 s = Small이 마지막으로 본 프레임. Base+ 는 s+1 부터 이어받는다.
객체 구간 [처음 보인 프레임, 영상 끝]의 25 / 50 / 75% 지점 (settings.SWITCH_FRACTIONS).
전환 이름은 비율 숫자 ("25", "50", "75") — 결과 줄에서 쓴다.
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


def object_exclusion(obj):
    """모든 전환/방법/회차에 공통으로 적용하는 객체 제외 사유. 정확히 8도 제외한다."""
    start, end = obj['start'], obj['end']
    expected = switch_points(start, end)
    if obj.get('switches') != expected:
        return {'reason': 'invalid_switch_schedule', 'expected_switches': expected}
    invalid = [dict(sw, pre_switch_frames=sw['frame'] - start)
               for sw in expected
               if sw['frame'] - start <= settings.MIN_TRACK_FRAMES
               or not start < sw['frame'] < end]
    if invalid:
        return {'reason': 'insufficient_pre_switch_frames', 'invalid_switches': invalid,
                'exclude_if_pre_switch_frames_lte': settings.MIN_TRACK_FRAMES,
                'scope': 'all_switches_methods_runs'}
    return None
