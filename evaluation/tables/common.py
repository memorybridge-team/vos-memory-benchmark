"""표 두 개(주 / 추가)가 같이 쓰는 것: 숫자 모양, 마크다운 표, 비교군별 줄 고르기."""

from __future__ import annotations

import settings
from evaluation.scoring.main_metrics import mean, video_means


def fmt(value, digits=1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def markdown(header: list[str], lines: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(line) + " |" for line in lines]
    return "\n".join(out)


def rows_of(rows: list[dict], baseline: str) -> list[dict]:
    return [r for r in rows if r["baseline"] == baseline]


def video_count(rows: list[dict]) -> str:
    return str(len({r["video"] for r in rows}))


def mean_over_videos(rows: list[dict], key: str, scale=1.0):
    """영상마다 평균 → 영상들의 평균 (VOS 벤치마크 관례)."""
    value = mean(video_means(rows, key).values())
    return None if value is None else value * scale


def main_metric(dataset: str) -> str:
    return "J" if dataset in settings.J_MAIN_DATASETS else "J&F"


def has_full_gt(rows: list[dict]) -> bool:
    """정답이 모든 프레임에 있어 여기서 채점한 데이터셋인가 (MOSEv2 valid 는 아님)."""
    return any(r.get("n_frames") for r in rows)
