"""[5] 결과 줄 → 표. 영상마다 회복률 → 평균 → 신뢰구간.

    python scripts/5_make_tables.py

읽는 것: outputs/records/*.jsonl, outputs/mosev2/server_rows.jsonl (있으면)
만드는 것 (outputs/tables/):
  main.md      주 표: 확정 비교군 10개 × 전환 A. J&F·J·회복률 (영상 전체 열 / 전환 뒤 열) + 비용
  extra.md     추가 표: [추가] 비교군, [추가] 지표, [추가] 조건별
  switch_b.md  전환 B 표 (재등장 직전)
점수는 100점 만점. "a [b, c]" = 평균 [95% 신뢰구간]. MOSEv2 와 비교할 때는 영상 전체 열만 쓴다.
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from benchmark import ci, records  # noqa: E402
from benchmark.methods import EXTRA, MAIN, METHODS  # noqa: E402
from benchmark.scoring import extra_metrics, extra_strata, mosev2_server  # noqa: E402
from benchmark.scoring.retention import retention_by_video, video_means  # noqa: E402


def load_rows() -> list[dict]:
    rows = []
    for path in sorted((Path(settings.OUTPUT_ROOT) / "records").glob("*.jsonl")):
        rows += records.read_rows(path)
    rows += records.read_rows(mosev2_server.mosev2_root() / "server_rows.jsonl")
    return rows


def fmt_ci(result, scale=1.0) -> str:
    if result is None:
        return "-"
    mean, lo, hi = (v * scale for v in result)
    return f"{mean:.1f} [{lo:.1f}, {hi:.1f}]"


def fmt(value, digits=1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def mean_over_videos(rows, key):
    values = list(video_means(rows, key).values())
    return sum(values) / len(values) if values else None


def markdown(header: list[str], lines: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(line) + " |" for line in lines]
    return "\n".join(out)


def method_table(rows: list[dict], methods) -> str:
    """비교군 × (영상 전체 / 전환 뒤) J&F·J·회복률 + 비용."""
    replay = [r for r in rows if r["method"] == "full_replay"]
    header = ["비교군", "J&F 전체", "J 전체", "회복률 전체",
              "J&F 전환 뒤", "J 전환 뒤", "회복률 전환 뒤",
              "다시 본 프레임", "첫 프레임까지(초)", "전환 뒤 프레임당(초)", "최대 VRAM(MB)", "영상 수"]
    lines = []
    for m in methods:
        mine = [r for r in rows if r["method"] == m.name]
        if not mine:
            continue
        ret_whole, _ = retention_by_video(mine, replay, "jf_whole")
        ret_post, _ = retention_by_video(mine, replay, "jf_post")
        j_whole, j_post = mean_over_videos(mine, "j_whole"), mean_over_videos(mine, "j_post")
        lines.append([
            m.label,
            fmt_ci(ci.bootstrap(video_means(mine, "jf_whole").values()), 100),
            fmt(None if j_whole is None else j_whole * 100),
            fmt_ci(ci.bootstrap(ret_whole.values())),
            fmt_ci(ci.bootstrap(video_means(mine, "jf_post").values()), 100),
            fmt(None if j_post is None else j_post * 100),
            fmt_ci(ci.bootstrap(ret_post.values())),
            fmt(mean_over_videos(mine, "reseen_frames")),
            fmt(mean_over_videos(mine, "seconds_to_first_frame"), 3),
            fmt(mean_over_videos(mine, "seconds_per_frame_after"), 3),
            fmt(mean_over_videos(mine, "peak_vram_mb"), 0),
            str(len({r["video"] for r in mine})),
        ])
    return markdown(header, lines)


def extra_metric_table(rows: list[dict]) -> str:
    names = {
        "extra_jf_at_n": f"J&F@{settings.EXTRA_JF_AT}",
        "extra_switch_shock": "switch shock",
        "extra_id_switch_rate": "ID 뒤바뀜 비율",
        "extra_recovery_frames": "회복 지연(프레임)",
        "extra_absent_false_alarm": "부재 오검출 비율",
        "extra_failure": "실패 비율",
    }
    scale = {"extra_jf_at_n": 100, "extra_switch_shock": 100, "extra_id_switch_rate": 100,
             "extra_recovery_frames": 1, "extra_absent_false_alarm": 100, "extra_failure": 100}
    lines = []
    for m in METHODS:
        mine = [r for r in rows if r["method"] == m.name]
        if not mine:
            continue
        cells = []
        for key in extra_metrics.KEYS:
            value = mean_over_videos(mine, key)
            cells.append(fmt(None if value is None else value * scale[key]))
        lines.append([m.label] + cells)
    return markdown(["비교군"] + [names[k] for k in extra_metrics.KEYS], lines)


def strata_tables(rows: list[dict]) -> str:
    parts = []
    for key in extra_strata.KEYS:
        chosen = [r for r in rows if r.get(key)]
        replay = [r for r in chosen if r["method"] == "full_replay"]
        lines = []
        for m in METHODS:
            mine = [r for r in chosen if r["method"] == m.name]
            if not mine:
                continue
            ratios, _ = retention_by_video(mine, replay, "jf_post")
            lines.append([m.label,
                          fmt_ci(ci.bootstrap(video_means(mine, "jf_post").values()), 100),
                          fmt_ci(ci.bootstrap(ratios.values())),
                          str(len({r["video"] for r in mine}))])
        parts.append(f"#### {extra_strata.LABELS[key]}\n\n"
                     + (markdown(["비교군", "J&F 전환 뒤", "회복률 전환 뒤", "영상 수"], lines)
                        if lines else "(해당 없음)"))
    return "\n\n".join(parts)


def main():
    rows = load_rows()
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r.get("part"))].append(r)

    main_md = ["# 주 표 — 확정 비교군 10개, 전환 A (25/50/75%)\n",
               "점수 100점 만점, `평균 [95% 구간]`. 회복률 = 영상마다 (방법 ÷ Full Replay × 100) 의 평균.",
               "MOSEv2 valid 는 서버 점수라 영상 전체 열만 있음. 다른 데이터셋과 비교할 때는 영상 전체 열을 쓴다.\n"]
    extra_md = ["# 추가 표 — 확정 표에 없는 것\n", "무엇이고 왜 넣었는지는 docs/EXTRAS.md.\n"]
    b_md = ["# [추가] 전환 B — 객체가 다시 나타나기 직전에 넘김\n"]

    for (dataset, part), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        title = dataset + (f" ({part})" if part else "")
        a_rows = [r for r in group if r["switch_set"] == "A"]
        b_rows = [r for r in group if r["switch_set"] == "B"]

        main_md += [f"## {title}\n", method_table(a_rows, MAIN), ""]

        extra_md += [f"## {title}\n",
                     "### [추가] 비교군 (전환 A)\n", method_table(a_rows, EXTRA), "",
                     "### [추가] 지표 (전환 A, 전환 뒤 구간, 영상 평균)\n", extra_metric_table(a_rows), "",
                     "### [추가] 조건별 (전환 A)\n", strata_tables(a_rows), ""]

        if b_rows:
            b_md += [f"## {title}\n", method_table(b_rows, METHODS), ""]

    out_dir = Path(settings.OUTPUT_ROOT) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in (("main.md", main_md), ("extra.md", extra_md), ("switch_b.md", b_md)):
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"저장: {out_dir / name}")


if __name__ == "__main__":
    main()
