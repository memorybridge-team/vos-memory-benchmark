"""전환 후 경과 프레임별 J·J&F. 각 시간점에서 객체 평균 → 영상 평균."""

from __future__ import annotations

from collections import defaultdict

from evaluation.methods import METHODS
from evaluation.scoring.main_metrics import mean

CSV_FIELDS = ["dataset", "switch_name", "baseline", "frames_after_switch",
              "j", "jf", "video_count", "object_count"]


def build(rows: list[dict]) -> list[dict]:
    values = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for point in row["frame_scores"]:
            key = (row["dataset"], row["switch_name"], row["baseline"], point["frames_after_switch"])
            values[key][row["video"]].append(point)

    order = {method.name: i for i, method in enumerate(METHODS)}
    out = []
    for key in sorted(values, key=lambda k: (k[0], int(k[1]), order[k[2]], k[3])):
        videos = values[key]
        out.append({
            "dataset": key[0], "switch_name": key[1], "baseline": key[2],
            "frames_after_switch": key[3],
            "j": mean(mean(p["j"] for p in points) for points in videos.values()) * 100,
            "jf": mean(mean(p["jf"] for p in points) for points in videos.values()) * 100,
            "video_count": len(videos),
            "object_count": sum(len(points) for points in videos.values()),
        })
    return out
