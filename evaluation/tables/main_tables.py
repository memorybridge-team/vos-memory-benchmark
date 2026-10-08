"""비교군 5개 + 본 모델의 전환별 점수, 기존 회복률, 전환 비용과 실패비율."""

from __future__ import annotations

import settings
from baseline import MAIN
from evaluation.scoring.main_metrics import retention
from evaluation.tables.common import fmt, main_metric, markdown, mean_over_videos, rows_of, video_count
from translator import MODEL

HEADER = ["방법", "J", "J&F", "회복률 J", "회복률 J&F",
          "전환시간(초)", "전환 GPU 메모리(MB)", "실패 비율(%)", "영상 수"]


def baseline_table(rows: list[dict], baselines) -> str:
    replay = rows_of(rows, "full_replay")
    lines = []
    for m in baselines:
        mine = rows_of(rows, m.name)
        if not mine:
            continue
        lines.append([
            m.label,
            fmt(mean_over_videos(mine, "j", 100)),
            fmt(mean_over_videos(mine, "jf", 100)),
            fmt(retention(mine, replay, "j")),
            fmt(retention(mine, replay, "jf")),
            fmt(mean_over_videos(mine, "switch_seconds"), 3),
            fmt(mean_over_videos(mine, "switch_gpu_mb"), 1),
            fmt(mean_over_videos(mine, "failure_rate", 100)),
            video_count(mine),
        ])
    return markdown(HEADER, lines)


def build(groups: dict) -> list[str]:
    out = ["# 평가표 — 비교군 5개 + 본 모델, 전환 시점 50/75%\n",
           "점수는 100점 만점, 객체별 평균을 영상별로 모은 뒤 영상 평균을 보고한다.",
           "J·J&F와 실패비율은 전환 뒤 정답에 객체가 보이는 프레임 기준.",
           "회복률은 기존 정의를 유지하며 복원율은 추후 논의한다.",
           "전환시간과 GPU 메모리는 준비 및 s까지의 replay 구간만 포함한다.\n"]
    for dataset, rows in groups.items():
        out += [f"## {dataset}\n", f"주 지표: {main_metric(dataset)}\n"]
        for fraction in settings.SWITCH_FRACTIONS:
            name = str(round(fraction * 100))
            mine = [r for r in rows if r["switch_name"] == name]
            if mine:
                out += [f"### 전환 {name}%\n", baseline_table(mine, MAIN + [MODEL]), ""]
    return out
