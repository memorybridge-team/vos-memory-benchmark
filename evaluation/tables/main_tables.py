"""비교군 5개 + 본 모델의 전환별 점수, 프레임별 회복률, 전환 비용과 실패비율."""

from __future__ import annotations

import settings
from baseline import MAIN
from evaluation.tables.summaries import format_stats, metric_stats
from evaluation.tables.common import main_metric, markdown, rows_of
from translator import MODEL

HEADER = ["방법", "J", "J&F", "회복률 J", "회복률 J&F",
          "전환시간(초)", "전환 GPU 메모리(MB)", "실패 비율(%)", "영상 수", "회차 수"]


def baseline_table(rows: list[dict], baselines) -> str:
    lines = []
    for m in baselines:
        mine = rows_of(rows, m.name)
        if not mine:
            continue
        lines.append([
            m.label,
            format_stats(mine, "j", 100),
            format_stats(mine, "jf", 100),
            format_stats(mine, "recovery_j"),
            format_stats(mine, "recovery_jf"),
            format_stats(mine, "switch_seconds", digits=3),
            format_stats(mine, "switch_gpu_mb"),
            format_stats(mine, "failure_rate", 100),
            str(metric_stats(mine, "j")["video_count"]),
            str(metric_stats(mine, "j")["run_count"]),
        ])
    return markdown(HEADER, lines)


def build(groups: dict) -> list[str]:
    out = ["# 평가표 — 비교군 5개 + 본 모델, 전환 시점 50/75%\n",
           "점수는 100점 만점, 객체별 평균을 영상별로 모은 뒤 영상 평균을 보고한다.",
           "J·J&F와 실패비율은 전환 뒤 정답에 객체가 보이는 프레임 기준.",
           "회복률은 동일 영상·객체·프레임의 방법/Native 반복 기준 × 100이다. 기본 기준은 중앙값이며 평균으로도 재집계할 수 있다.",
           "회차별 객체→영상 평균 후 반복 평균 ± 표본 표준편차(ddof=1)를 보고한다.",
           "한 회차만 있으면 표준편차는 없으며, 부분 결과는 회차 간 공통 유효 사례 기준이다.",
           "Native 기준은 선택한 모든 회차가 완료된 프레임에서만 확정한다. 부족하면 회복률은 N/A이다.",
           "Native의 실제 회차 점수도 기준으로 나누므로 Native 회복률 평균이 항상 100%인 것은 아니다.",
           "성능 실패는 포함한다. Native=0인 프레임은 해당 회복률만 N/A이며 원점수·실패비율은 유지한다.",
           "전환시간과 GPU 메모리는 준비 및 s까지의 replay 구간만 포함한다.\n"]
    for dataset, rows in groups.items():
        out += [f"## {dataset}\n", f"주 지표: {main_metric(dataset)}\n"]
        for fraction in settings.SWITCH_FRACTIONS:
            name = str(round(fraction * 100))
            mine = [r for r in rows if r["switch_name"] == name]
            if mine:
                out += [f"### 전환 {name}%\n", baseline_table(mine, MAIN + [MODEL]), ""]
    return out
