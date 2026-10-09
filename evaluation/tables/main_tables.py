"""비교군 5개 + 본 모델의 전환별 점수, 프레임별 회복률, 전환 비용과 실패비율."""

from __future__ import annotations

import settings
from baseline import MAIN
from evaluation.tables.summaries import format_stats, metric_stats
from evaluation.tables.common import main_metric, markdown, rows_of
from translator import MODEL

HEADER = ["방법", "J", "J&F", "전환 전 회복률 J", "전환 전 회복률 J&F", "전환 후 회복률 J", "전환 후 회복률 J&F", "복원율 R² (spatial)", "복원율 R² (pointer)",
          "전환시간(초)", "전환 GPU 메모리(MB)", "실패 비율(%)", "영상 수", "회차 수", "R² 유효 객체 spatial/pointer"]


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
            format_stats(mine, "pre_recovery_j"),
            format_stats(mine, "pre_recovery_jf"),
            format_stats(mine, "post_recovery_j"),
            format_stats(mine, "post_recovery_jf"),
            format_stats(mine, "r2_maskmem_features", digits=3),
            format_stats(mine, "r2_obj_ptr", digits=3),
            format_stats(mine, "switch_seconds", digits=3),
            format_stats(mine, "switch_gpu_mb"),
            format_stats(mine, "failure_rate", 100),
            str(metric_stats(mine, "j")["video_count"]),
            str(metric_stats(mine, "j")["run_count"]),
            f"{metric_stats(mine, 'r2_maskmem_features')['object_count']}/{metric_stats(mine, 'r2_obj_ptr')['object_count']}",
        ])
    return markdown(HEADER, lines)


def build(groups: dict) -> list[str]:
    out = ["# 평가표 — 비교군 5개 + 본 모델, 전환 시점 25/50/75%\n",
           "점수는 100점 만점, 객체별 평균을 영상별로 모은 뒤 영상 평균을 보고한다.",
           "J·J&F와 실패비율은 전환 뒤 정답에 객체가 보이는 프레임 기준.",
           "전환 전 값은 Native를 제외하면 같은 Small 실행이며 독립적인 방법 비교가 아니다. 처음 정답 프롬프트 프레임도 포함한다.",
           "회복률은 객체별로 구간 평균 점수 / 같은 프레임의 Native 기준 평균 × 100이다. Native 기준은 프레임별 반복 중앙값이며 평균으로도 재집계할 수 있다.",
           "전환 전은 객체 최초 등장~s, 전환 후는 s+1~끝이다. Native를 제외한 방법의 전환 전은 동일 Small 예측이고, Native 행은 자체 전환 전 예측이다.",
           "회차별 객체→영상 평균 후 반복 평균 ± 표본 표준편차(ddof=1)를 보고한다.",
           "한 회차만 있으면 표준편차는 없으며, 부분 결과는 회차 간 공통 유효 사례 기준이다.",
           "Native 기준은 선택한 모든 회차가 완료된 프레임에서만 확정한다. 부족하면 회복률은 N/A이다.",
           "Native의 실제 회차 점수도 기준으로 나누므로 Native 회복률 평균이 항상 100%인 것은 아니다.",
           "성능 실패와 Native=0 프레임도 구간 평균에 포함한다. 구간 Native 평균이 0이면 구간 회복률은 N/A이다. 누락된 Native 프레임은 분자·분모에서 함께 제외한다.",
           "복원율은 준비된 Target 기억 전체의 R²을 필드별로 계산하고 객체→영상→회차 순서로 평균한다. 칸별 R²의 단순 평균은 아니다.",
           "Target 칸의 대응 Native가 누락되면 해당 전체 R²은 N/A이다. 유효 객체 수는 지표마다 다를 수 있다.",
           "전환시간과 GPU 메모리는 준비 및 s까지의 replay 구간만 포함한다.\n"]
    for dataset, rows in groups.items():
        out += [f"## {dataset}\n", f"주 지표: {main_metric(dataset)}\n"]
        for fraction in settings.SWITCH_FRACTIONS:
            name = str(round(fraction * 100))
            mine = [r for r in rows if r["switch_name"] == name]
            if mine:
                out += [f"### 전환 {name}%\n", baseline_table(mine, MAIN + [MODEL]), ""]
    return out
