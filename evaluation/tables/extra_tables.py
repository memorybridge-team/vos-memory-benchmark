"""[추가] 표: 확정 표에 없는 것. 무엇이고 왜 넣었는지는 docs/EXTRAS.md.

데이터셋마다
  비용 세부          전환 뒤 프레임당 시간, GPU 메모리 최고치
  출력 일치도        방법 마스크 vs Full Replay 마스크 IoU (MOSEv2 valid 에도 있음)
  진단 비교군        reset, recent_K_only (↔ Original+Replay-K)
  ── 아래는 정답이 모든 프레임에 있는 데이터셋만 (MOSEv2 valid 없음) ──
  실패 분석          ID 뒤바뀜 비율, 실패 비율
  drift              전환 뒤 경과 구간별 (방법 − Full Replay) J&F + 곡선 PNG
  공식 라벨별        라벨마다 회복률 (MOSEv2·PUMaVOS 는 라벨이 없어 빈 표)
  입력 길이별        전환 전 Small 이 본 프레임 수 구간마다 회복률
마지막에 여러 데이터셋에서 같은 뜻인 라벨을 합친 표 (평가 데이터셋만).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import settings
from baseline import BASELINES, BY_NAME, MAIN
from evaluation.scoring import extra_groups
from evaluation.scoring.main_metrics import mean, retention
from evaluation.tables.common import (fmt, has_full_gt, markdown, mean_over_videos, rows_of,
                                     video_count)
from evaluation.tables.main_tables import baseline_table

# (열 이름, 결과 줄 열, 곱할 값, 소수 자리)
COST = [("전환 뒤 프레임당(초)", "seconds_per_frame_after", 1, 3), ("최대 GPU 메모리(MB)", "peak_vram_mb", 1, 0)]
FAILURE = [("ID 뒤바뀜 비율(%)", "extra_id_switch_rate", 100, 1), ("실패 비율(%)", "extra_failure", 100, 1)]
AGREEMENT = [("출력 일치도(%)", "extra_agreement", 100, 1)]


def column_table(rows: list[dict], columns) -> str:
    """비교군마다 열들의 영상 평균."""
    lines = []
    for m in BASELINES:
        mine = rows_of(rows, m.name)
        if mine:
            lines.append([m.label] + [fmt(mean_over_videos(mine, key, scale), digits)
                                      for _, key, scale, digits in columns] + [video_count(mine)])
    return markdown(["비교군"] + [name for name, *_ in columns] + ["영상 수"], lines)


def diagnostic_baselines() -> list:
    """reset, recent_K_only, 그리고 비교 상대 Original+Replay-K."""
    names = ["reset", "recent_k_only", f"original_replay_{settings.EXTRA_RECENT_K}"]
    return [BY_NAME[n] for n in names if n in BY_NAME]


def group_tables(rows: list[dict], groups: list[str], groups_of_row) -> str:
    """묶음마다 주 비교군의 회복률. 행 = 비교군, 열 = 묶음. J&F 표와 J 표 두 개."""
    if not groups:
        return "(없음)"
    chosen = {g: [r for r in rows if g in groups_of_row(r)] for g in groups}
    header = ["비교군"] + [f"{g} (영상 {video_count(chosen[g])})" for g in groups]
    parts = []
    for key, name in (("jf", "J&F"), ("j", "J")):
        lines = [[m.label] + [fmt(retention(rows_of(chosen[g], m.name), rows_of(chosen[g], "full_replay"), key))
                              for g in groups]
                 for m in MAIN]
        parts.append(f"회복률 {name}\n\n" + markdown(header, lines))
    return "\n\n".join(parts)


def drift_curves(rows: list[dict]) -> dict[str, list]:
    """비교군 → 구간마다 (방법 − 같은 전환의 Full Replay) J&F 차이의 영상 평균 (100점 만점)."""
    replay = {(r["video"], r["object"], r["switch_name"]): r["extra_drift_jf"]
              for r in rows_of(rows, "full_replay")}
    n_bins = len(settings.EXTRA_DRIFT_BINS)
    curves = {}
    for m in BASELINES:
        if m.name == "full_replay" or not rows_of(rows, m.name):
            continue
        per_video = defaultdict(lambda: [[] for _ in range(n_bins)])   # 영상 → 구간 → 차이들
        for r in rows_of(rows, m.name):
            base = replay[(r["video"], r["object"], r["switch_name"])]
            for i, (a, b) in enumerate(zip(r["extra_drift_jf"], base)):
                if a is not None and b is not None:
                    per_video[r["video"]][i].append((a - b) * 100)
        curves[m.label] = [mean(mean(bins[i]) for bins in per_video.values()) for i in range(n_bins)]
    return curves


def drift_section(rows: list[dict], png: Path, title: str) -> str:
    curves = drift_curves(rows)
    names = extra_groups.bin_names(settings.EXTRA_DRIFT_BINS)
    lines = [[label] + [fmt(v) for v in values] for label, values in curves.items()]
    _plot_drift(curves, names, png, title)
    return (markdown(["비교군 \\ 전환 뒤 프레임"] + names, lines)
            + f"\n\n![drift]({png.name})")


def _plot_drift(curves: dict, names: list[str], png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_prop_cycle(color=plt.cm.tab20.colors)    # 비교군이 10개를 넘어도 색이 겹치지 않게
    for label, values in curves.items():
        ax.plot(names, [float("nan") if v is None else v for v in values], marker="o", label=label)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(title=f"drift — {title}", xlabel="frames after switch", ylabel="J&F − Full Replay")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=120)
    plt.close(fig)


def dataset_section(rows: list[dict], title: str, raw_labels, png: Path) -> list[str]:
    out = [f"## {title}\n",
           "### 비용 세부\n", column_table(rows, COST), "",
           "### 출력 일치도 (Full Replay 마스크와 IoU)\n", column_table(rows, AGREEMENT), "",
           f"### 진단 비교군 — recent_{settings.EXTRA_RECENT_K}_only ↔ "
           f"Original+Replay-{settings.EXTRA_RECENT_K}\n",
           baseline_table(rows, diagnostic_baselines()), ""]
    if not has_full_gt(rows):
        return out + ["(정답이 첫 프레임뿐이라 실패 분석·drift·조건별·입력 길이는 없음)", ""]

    length_bins = [b for b in extra_groups.bin_names(settings.EXTRA_INPUT_LENGTH_BINS)
                   if any(extra_groups.input_length_bin(r) == b for r in rows)]
    labels = sorted({label for r in rows for label in raw_labels(r)})
    out += ["### 실패 분석 (전환 뒤 보이는 프레임)\n", column_table(rows, FAILURE), "",
            "### drift (방법 − Full Replay J&F)\n", drift_section(rows, png, title), "",
            "### 공식 라벨별\n", group_tables(rows, labels, raw_labels), "",
            "### 입력 길이별 (전환 전 Small 이 본 프레임 수)\n",
            group_tables(rows, length_bins, lambda r: [extra_groups.input_length_bin(r)]), ""]
    return out


def build(groups: dict, object_labels: dict, out_dir: Path) -> list[str]:
    """groups = {(데이터셋, part): 결과 줄}, object_labels = {(데이터셋, 영상, 객체): [라벨]} → extra.md 의 줄들."""
    def raw_labels(r):
        return object_labels.get((r["dataset"], r["video"], r["object"]), [])

    out = ["# 추가 표 — 확정 표에 없는 것\n", "무엇이고 왜 넣었는지는 docs/EXTRAS.md.\n"]
    for (dataset, part), rows in groups.items():
        name = dataset + (f"_{part}" if part else "")
        title = dataset + (f" ({part})" if part else "")
        out += dataset_section(rows, title, raw_labels, out_dir / f"drift_{name}.png")

    # 여러 데이터셋에서 같은 뜻인 라벨을 합친 표 (평가 데이터셋만, train 은 뺌).
    # 영상 이름이 데이터셋끼리 겹칠 수 있어 "데이터셋/영상" 으로 구분한다.
    pooled = []
    for (dataset, part), rows in groups.items():
        for r in rows:
            if part is not None:
                continue
            merged = sorted({n for label in raw_labels(r) if (n := extra_groups.common_name(dataset, label))})
            if merged:
                pooled.append(dict(r, video=f"{dataset}/{r['video']}", merged_labels=merged))
    merged_names = sorted({n for r in pooled for n in r["merged_labels"]})
    out += ["## 여러 데이터셋 합친 라벨\n",
            "합치는 규칙은 evaluation/scoring/extra_groups.py 의 SAME_AS.\n",
            group_tables(pooled, merged_names, lambda r: r["merged_labels"]), ""]
    return out
