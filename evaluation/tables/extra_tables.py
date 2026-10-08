"""난이도 유형별 J·J&F·실패비율과 기존 회복률. 라벨은 영상 또는 객체 전체에 붙는다."""

from __future__ import annotations

import settings
from baseline import MAIN
from evaluation.scoring import extra_groups
from evaluation.scoring.main_metrics import retention
from evaluation.tables.common import fmt, markdown, mean_over_videos, rows_of, video_count
from translator import MODEL


def group_tables(rows: list[dict], groups: list[str], groups_of_row) -> str:
    if not groups:
        return "(난이도 라벨 없음)"
    parts = []
    for group in groups:
        chosen = [r for r in rows if group in groups_of_row(r)]
        replay = rows_of(chosen, "full_replay")
        lines = []
        for method in MAIN + [MODEL]:
            mine = rows_of(chosen, method.name)
            if mine:
                lines.append([
                    method.label,
                    fmt(mean_over_videos(mine, "j", 100)),
                    fmt(mean_over_videos(mine, "jf", 100)),
                    fmt(mean_over_videos(mine, "failure_rate", 100)),
                    fmt(retention(mine, replay, "j")),
                    fmt(retention(mine, replay, "jf")),
                    video_count(mine),
                ])
        parts += [f"#### {group}", "",
                  markdown(["방법", "J", "J&F", "실패 비율(%)", "회복률 J", "회복률 J&F", "영상 수"], lines), ""]
    return "\n".join(parts)


def switch_sections(rows: list[dict], groups_of_row) -> list[str]:
    out = []
    for fraction in settings.SWITCH_FRACTIONS:
        name = str(round(fraction * 100))
        mine = [r for r in rows if r["switch_name"] == name]
        groups = sorted({group for r in mine for group in groups_of_row(r)})
        if mine:
            out += [f"### 전환 {name}%\n", group_tables(mine, groups, groups_of_row), ""]
    return out


def build(groups: dict, object_labels: dict) -> list[str]:
    def raw_labels(row):
        return object_labels.get((row["dataset"], row["video"], row["object"]), [])

    out = ["# 난이도 유형별 성능\n",
           "영상·객체 전체의 라벨 기준이며 전환 뒤에 해당 사건이 발생했는지는 구분하지 않는다.\n"]
    for dataset, rows in groups.items():
        out += [f"## {dataset} 공식 라벨\n"] + switch_sections(rows, raw_labels)

    pooled = []
    for dataset, rows in groups.items():
        for row in rows:
            merged = sorted({name for label in raw_labels(row)
                             if (name := extra_groups.common_name(dataset, label))})
            if merged:
                pooled.append(dict(row, video=f"{dataset}/{row['video']}", difficulty_types=merged))
    out += ["## 여러 데이터셋의 공통 난이도 유형\n"]
    out += switch_sections(pooled, lambda row: row["difficulty_types"])
    return out
