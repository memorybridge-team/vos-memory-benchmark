"""[추가] 표: 확정 표에 없는 것. 무엇이고 왜 넣었는지는 docs/EXTRAS.md.

데이터셋마다
  비용 세부          전환 GPU 메모리 (전환 구간 최고치 − 전환 직전)
  출력 일치도        방법 마스크 vs Full Replay 마스크 IoU
  진단 비교군        reset, recent_K_only (↔ Original+Replay-K)
  실패 분석          실패 비율 (1 − J_Recall)
  공식 라벨별        라벨마다 회복률 (PUMaVOS 는 라벨이 없어 "(없음)")
마지막에 여러 데이터셋에서 같은 뜻인 라벨을 합친 표.
"""

from __future__ import annotations

import settings
from baseline import BY_NAME, MAIN
from evaluation.methods import METHODS
from evaluation.scoring import extra_groups
from evaluation.scoring.main_metrics import retention
from evaluation.tables.common import fmt, markdown, mean_over_videos, rows_of, video_count
from evaluation.tables.main_tables import baseline_table
from translator import MODEL

# (열 이름, 결과 줄 열, 곱할 값, 소수 자리)
COST = [("전환 GPU 메모리(MB)", "extra_switch_gpu_mb", 1, 0)]
FAILURE = [("실패 비율(%)", "extra_failure_rate", 100, 1)]
AGREEMENT = [("출력 일치도(%)", "extra_agreement", 100, 1)]


def column_table(rows: list[dict], columns) -> str:
    """방법마다 열들의 영상 평균."""
    lines = []
    for m in METHODS:
        mine = rows_of(rows, m.name)
        if mine:
            lines.append([m.label] + [fmt(mean_over_videos(mine, key, scale), digits)
                                      for _, key, scale, digits in columns] + [video_count(mine)])
    return markdown(["방법"] + [name for name, *_ in columns] + ["영상 수"], lines)


def diagnostic_baselines() -> list:
    """reset, recent_K_only, 그리고 비교 상대 Original+Replay-K."""
    names = ["reset", "recent_k_only", f"original_replay_{settings.EXTRA_RECENT_K}"]
    return [BY_NAME[n] for n in names if n in BY_NAME]


def group_tables(rows: list[dict], groups: list[str], groups_of_row) -> str:
    """묶음마다 주 비교군·본 모델의 회복률. 행 = 방법, 열 = 묶음. J&F 표와 J 표 두 개."""
    if not groups:
        return "(없음)"
    chosen = {g: [r for r in rows if g in groups_of_row(r)] for g in groups}
    header = ["방법"] + [f"{g} (영상 {video_count(chosen[g])})" for g in groups]
    parts = []
    for key, name in (("jf", "J&F"), ("j", "J")):
        lines = [[m.label] + [fmt(retention(rows_of(chosen[g], m.name), rows_of(chosen[g], "full_replay"), key))
                              for g in groups]
                 for m in MAIN + [MODEL]]
        parts.append(f"회복률 {name}\n\n" + markdown(header, lines))
    return "\n\n".join(parts)


def dataset_section(rows: list[dict], dataset: str, raw_labels) -> list[str]:
    labels = sorted({label for r in rows for label in raw_labels(r)})
    return [f"## {dataset}\n",
            "### 비용 세부\n", column_table(rows, COST), "",
            "### 출력 일치도 (Full Replay 마스크와 IoU)\n", column_table(rows, AGREEMENT), "",
            f"### 진단 비교군 — recent_{settings.EXTRA_RECENT_K}_only ↔ "
            f"Original+Replay-{settings.EXTRA_RECENT_K}\n",
            baseline_table(rows, diagnostic_baselines()), "",
            "### 실패 분석 (전환 뒤 보이는 프레임)\n", column_table(rows, FAILURE), "",
            "### 공식 라벨별\n", group_tables(rows, labels, raw_labels), ""]


def build(groups: dict, object_labels: dict) -> list[str]:
    """groups = {데이터셋: 결과 줄}, object_labels = {(데이터셋, 영상, 객체): [라벨]} → extra.md 의 줄들."""
    def raw_labels(r):
        return object_labels.get((r["dataset"], r["video"], r["object"]), [])

    out = ["# 추가 표 — 확정 표에 없는 것\n", "무엇이고 왜 넣었는지는 docs/EXTRAS.md.\n"]
    for dataset, rows in groups.items():
        out += dataset_section(rows, dataset, raw_labels)

    # 여러 데이터셋에서 같은 뜻인 라벨을 합친 표.
    # 영상 이름이 데이터셋끼리 겹칠 수 있어 "데이터셋/영상" 으로 구분한다.
    pooled = []
    for dataset, rows in groups.items():
        for r in rows:
            merged = sorted({n for label in raw_labels(r) if (n := extra_groups.common_name(dataset, label))})
            if merged:
                pooled.append(dict(r, video=f"{dataset}/{r['video']}", merged_labels=merged))
    merged_names = sorted({n for r in pooled for n in r["merged_labels"]})
    out += ["## 여러 데이터셋 합친 라벨\n",
            "합치는 규칙은 evaluation/scoring/extra_groups.py 의 SAME_AS.\n",
            group_tables(pooled, merged_names, lambda r: r["merged_labels"]), ""]
    return out
