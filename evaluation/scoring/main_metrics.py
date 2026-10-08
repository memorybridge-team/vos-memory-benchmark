"""J·J&F 구간 평균과 실패비율. 프레임별 회복률은 recovery.py."""
from __future__ import annotations

from collections import defaultdict

import settings


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def score_columns(scores: list) -> dict:
    """전환 뒤 보이는 프레임들의 FrameScore → j, jf, n_frames."""
    if not scores:
        return {"j": None, "jf": None, "n_frames": 0}
    j = mean(s.j for s in scores)
    f = mean(s.f for s in scores)
    return {"j": j, "jf": (j + f) / 2, "n_frames": len(scores)}



def failure_rate(scores: list):
    """실패 비율 = 1 − mean(J > 0.5). J=0.5는 성공에 포함하지 않는다."""
    recall = mean(s.j > settings.RECALL_J for s in scores)
    return None if recall is None else 1 - recall


def video_means(rows: list[dict], key: str) -> dict[str, float]:
    """영상 → 그 영상 줄들(객체·전환 시점)의 key 평균. 값이 없는 줄은 건너뛴다."""
    values = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            values[r["video"]].append(r[key])
    return {video: sum(v) / len(v) for video, v in values.items()}
