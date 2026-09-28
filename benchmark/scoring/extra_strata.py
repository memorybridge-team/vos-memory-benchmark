"""[추가] 조건별 분류. 정답만 보고 (영상, 객체, 전환 시점) 마다 어떤 어려운 상황인지 표시한다.
결과 열 이름은 모두 extra_stratum_ 로 시작. 전환 뒤 구간만 본다.

extra_stratum_occlusion  가림: 전환 뒤에 객체가 사라졌다가 다시 나타남
extra_stratum_crossing   교차: 전환 뒤에 다른 객체와 상자가 겹침
extra_stratum_small      작은 객체: 보이는 넓이의 중앙값이 화면의 EXTRA_SMALL_AREA 보다 작음
extra_stratum_fast       빠른 움직임: 프레임당 중심 이동의 중앙값이 대각선의 EXTRA_FAST_MOTION 보다 큼
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import settings
from benchmark.data.common import object_ids

KEYS = ("extra_stratum_occlusion", "extra_stratum_crossing",
        "extra_stratum_small", "extra_stratum_fast")
LABELS = {
    "extra_stratum_occlusion": "가림 (사라졌다 다시 나타남)",
    "extra_stratum_crossing": "교차 (다른 객체와 겹침)",
    "extra_stratum_small": "작은 객체",
    "extra_stratum_fast": "빠른 움직임",
}


@dataclass
class GtFrame:
    visible: bool
    area: float = 0.0                 # 화면 대비 넓이
    center: tuple = (0.0, 0.0)
    box: tuple | None = None          # (y0, x0, y1, x1)
    other_boxes: tuple = ()


def _box(mask):
    ys, xs = np.nonzero(mask)
    return ys.min(), xs.min(), ys.max() + 1, xs.max() + 1


def _box_iou(a, b) -> float:
    y0, x0 = max(a[0], b[0]), max(a[1], b[1])
    y1, x1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, y1 - y0) * max(0, x1 - x0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


def gt_profile(video, obj_id: int, start: int, end: int) -> dict[int, GtFrame]:
    """정답 프레임마다 이 객체의 보임·넓이·중심·상자, 다른 객체들의 상자."""
    height, width = video.size
    profile = {}
    for frame in sorted(f for f in video.mask_paths if start < f <= end):
        labels, _ = video.read_labels(frame)
        own = labels == obj_id
        if not own.any():
            profile[frame] = GtFrame(False)
            continue
        ys, xs = np.nonzero(own)
        others = tuple(_box(labels == o) for o in object_ids(labels) if o != obj_id)
        profile[frame] = GtFrame(True, own.sum() / (height * width),
                                 (ys.mean(), xs.mean()), _box(own), others)
    return profile


def classify(profile: dict[int, GtFrame], video, switch_frame: int) -> dict[str, bool]:
    frames = [f for f in sorted(profile) if f > switch_frame]
    seen = [f for f in frames if profile[f].visible]
    height, width = video.size
    diag = np.hypot(height, width)

    occlusion = False
    if seen:
        gone = [f for f in frames if not profile[f].visible and f > seen[0]]
        occlusion = any(g < f for g in gone for f in seen)

    crossing = any(_box_iou(profile[f].box, other) > settings.EXTRA_CROSSING_BOX_IOU
                   for f in seen for other in profile[f].other_boxes)

    small = bool(seen) and np.median([profile[f].area for f in seen]) < settings.EXTRA_SMALL_AREA

    speeds = [np.hypot(*np.subtract(profile[b].center, profile[a].center)) / (b - a) / diag
              for a, b in zip(seen, seen[1:])]
    fast = bool(speeds) and np.median(speeds) > settings.EXTRA_FAST_MOTION

    return {"extra_stratum_occlusion": bool(occlusion), "extra_stratum_crossing": bool(crossing),
            "extra_stratum_small": bool(small), "extra_stratum_fast": bool(fast)}
