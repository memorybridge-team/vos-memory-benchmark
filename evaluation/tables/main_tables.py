"""[주] 표: 확정 비교군 9개 + 본 모델 × 전환 시점 25/50/75%.

열
  J, J&F                     전환 뒤, 객체가 보이는 프레임만
  회복률 J, 회복률 J&F         영상마다 방법 ÷ Full Replay × 100 → 평균
  격차 회복률 J, J&F           (방법 − Source-only) ÷ (Full Replay − Source-only) × 100, 영상 평균 점수로 한 번
  시간(초)                   Base+ 가 한 일 전부 (Full Replay = 처음 ~ 끝, 다른 비교군 = 넘기기 + s+1 ~ 끝)
점수는 100점 만점. 데이터셋마다 주 지표(VOST·M3VOS 는 J, 나머지는 J&F)를 제목 아래에 적는다.
"""

from __future__ import annotations

from baseline import MAIN
from evaluation.scoring.main_metrics import gap_retention, retention
from evaluation.tables.common import fmt, main_metric, markdown, mean_over_videos, rows_of, video_count
from translator import MODEL

HEADER = ["방법", "J", "J&F", "회복률 J", "회복률 J&F", "격차 회복률 J", "격차 회복률 J&F",
          "시간(초)", "영상 수"]


def baseline_table(rows: list[dict], baselines) -> str:
    replay, source = rows_of(rows, "full_replay"), rows_of(rows, "source_only")
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
            fmt(gap_retention(mine, source, replay, "j")),
            fmt(gap_retention(mine, source, replay, "jf")),
            fmt(mean_over_videos(mine, "seconds"), 2),
            video_count(mine),
        ])
    return markdown(HEADER, lines)


def build(groups: dict) -> list[str]:
    """groups = {데이터셋: 결과 줄} → main.md 의 줄들."""
    out = ["# 주 표 — 확정 비교군 9개 + 본 모델, 전환 시점 25/50/75%\n",
           "점수 100점 만점, 영상 평균. 회복률 = 영상마다 (방법 ÷ Full Replay × 100) 의 평균.",
           "격차 회복률 = (방법 − Source-only) ÷ (Full Replay − Source-only) × 100 (영상 평균 점수로 한 번).",
           "J, J&F 는 전환 뒤 객체가 보이는 프레임 기준.",
           "시간 = Base+ 가 한 일 전부 (Full Replay = 처음 ~ 끝, 다른 비교군 = 넘기기 + s+1 ~ 끝).\n"]
    for dataset, rows in groups.items():
        out += [f"## {dataset}\n",
                f"주 지표: {main_metric(dataset)}\n",
                baseline_table(rows, MAIN + [MODEL]), ""]
    return out
