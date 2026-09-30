"""[5] 결과 줄 → 표. 영상마다 회복률 → 평균.

    python scripts/5_make_tables.py

읽는 것: outputs/records/*.jsonl, outputs/mosev2/server_rows.jsonl (있으면)
만드는 것 (outputs/tables/):
  main.md      주 표: 확정 비교군 10개 × 전환 A. J&F·J·회복률 (영상 전체 열 / 전환 뒤 열) + 비용
  extra.md     추가 표: [추가] 비교군, [추가] 지표, [추가] 조건별, [추가] 공식 라벨별 (데이터셋별 + 합친 것)
  switch_b.md  전환 B 표 (재등장 직전)
점수는 100점 만점, 영상 평균 (VOS 벤치마크 관례대로 숫자 하나). MOSEv2 와 비교할 때는 영상 전체 열만 쓴다.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from benchmark import records  # noqa: E402
from benchmark.baselines import EXTRA, MAIN, BASELINES  # noqa: E402
from benchmark.scoring import extra_labels, extra_metrics, extra_strata, mosev2_server  # noqa: E402
from benchmark.scoring.retention import retention_by_video, video_means  # noqa: E402


def load_rows() -> list[dict]:
    rows = []
    for path in sorted((Path(settings.OUTPUT_ROOT) / "records").glob("*.jsonl")):
        rows += records.read_rows(path)
    rows += records.read_rows(mosev2_server.mosev2_root() / "server_rows.jsonl")
    return rows


def fmt(value, digits=1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def mean_over_videos(rows, key, scale=1.0):
    value = mean(video_means(rows, key).values())
    return None if value is None else value * scale


def markdown(header: list[str], lines: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(line) + " |" for line in lines]
    return "\n".join(out)


def baseline_table(rows: list[dict], baselines) -> str:
    """비교군 × (영상 전체 / 전환 뒤) J&F·J·회복률 + 비용."""
    replay = [r for r in rows if r["baseline"] == "full_replay"]
    header = ["비교군", "J&F 전체", "J 전체", "회복률 전체",
              "J&F 전환 뒤", "J 전환 뒤", "회복률 전환 뒤",
              "다시 본 프레임", "첫 프레임까지(초)", "전환 뒤 프레임당(초)", "최대 VRAM(MB)", "영상 수"]
    lines = []
    for m in baselines:
        mine = [r for r in rows if r["baseline"] == m.name]
        if not mine:
            continue
        ret_whole, _ = retention_by_video(mine, replay, "jf_whole")
        ret_post, _ = retention_by_video(mine, replay, "jf_post")
        lines.append([
            m.label,
            fmt(mean_over_videos(mine, "jf_whole", 100)),
            fmt(mean_over_videos(mine, "j_whole", 100)),
            fmt(mean(ret_whole.values())),
            fmt(mean_over_videos(mine, "jf_post", 100)),
            fmt(mean_over_videos(mine, "j_post", 100)),
            fmt(mean(ret_post.values())),
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
    for m in BASELINES:
        mine = [r for r in rows if r["baseline"] == m.name]
        if not mine:
            continue
        cells = []
        for key in extra_metrics.KEYS:
            cells.append(fmt(mean_over_videos(mine, key, scale[key])))
        lines.append([m.label] + cells)
    return markdown(["비교군"] + [names[k] for k in extra_metrics.KEYS], lines)


def strata_tables(rows: list[dict]) -> str:
    parts = []
    for key in extra_strata.KEYS:
        chosen = [r for r in rows if r.get(key)]
        replay = [r for r in chosen if r["baseline"] == "full_replay"]
        lines = []
        for m in BASELINES:
            mine = [r for r in chosen if r["baseline"] == m.name]
            if not mine:
                continue
            ratios, _ = retention_by_video(mine, replay, "jf_post")
            lines.append([m.label,
                          fmt(mean_over_videos(mine, "jf_post", 100)),
                          fmt(mean(ratios.values())),
                          str(len({r["video"] for r in mine}))])
        parts.append(f"#### {extra_strata.LABELS[key]}\n\n"
                     + (markdown(["비교군", "J&F 전환 뒤", "회복률 전환 뒤", "영상 수"], lines)
                        if lines else "(해당 없음)"))
    return "\n\n".join(parts)


def load_object_labels() -> dict:
    """[추가] (데이터셋, 영상, 객체) → 공식 라벨. 1_make_video_list.py 가 목록에 적어 둔 것."""
    labels = {}
    for path in sorted((Path(settings.OUTPUT_ROOT) / "lists").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data["videos"]:
            for obj in entry["objects"]:
                labels[(data["dataset"], entry["video"], obj["object"])] = obj.get("extra_labels", [])
    return labels


def label_table(rows: list[dict], labels_of_row) -> str:
    """[추가] 라벨마다 확정 비교군의 회복률 (전환 뒤 기준). 라벨이 붙은 영상만 모아 계산."""
    lines = []
    for label in sorted({label for r in rows for label in labels_of_row(r)}):
        chosen = [r for r in rows if label in labels_of_row(r)]
        replay = [r for r in chosen if r["baseline"] == "full_replay"]
        cells = []
        for m in MAIN:
            ratios, _ = retention_by_video([r for r in chosen if r["baseline"] == m.name],
                                           replay, "jf_post")
            cells.append(fmt(mean(ratios.values())))
        lines.append([label, str(len({r["video"] for r in chosen}))] + cells)
    if not lines:
        return "(라벨 없음)"
    return markdown(["라벨", "영상 수"] + [m.label for m in MAIN], lines)


def main():
    rows = load_rows()
    object_labels = load_object_labels()

    def raw_labels(r):
        return object_labels.get((r["dataset"], r["video"], r["object"]), [])

    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r.get("part"))].append(r)

    main_md = ["# 주 표 — 확정 비교군 10개, 전환 A (25/50/75%)\n",
               "점수 100점 만점, 영상 평균. 회복률 = 영상마다 (방법 ÷ Full Replay × 100) 의 평균.",
               "MOSEv2 valid 는 서버 점수라 영상 전체 열만 있음. 다른 데이터셋과 비교할 때는 영상 전체 열을 쓴다.\n"]
    extra_md = ["# 추가 표 — 확정 표에 없는 것\n", "무엇이고 왜 넣었는지는 docs/EXTRAS.md.\n"]
    b_md = ["# [추가] 전환 B — 객체가 다시 나타나기 직전에 넘김\n"]

    for (dataset, part), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        title = dataset + (f" ({part})" if part else "")
        a_rows = [r for r in group if r["switch_set"] == "A"]
        b_rows = [r for r in group if r["switch_set"] == "B"]

        main_md += [f"## {title}\n", baseline_table(a_rows, MAIN), ""]

        extra_md += [f"## {title}\n",
                     "### [추가] 비교군 (전환 A)\n", baseline_table(a_rows, EXTRA), "",
                     "### [추가] 지표 (전환 A, 전환 뒤 구간, 영상 평균)\n", extra_metric_table(a_rows), "",
                     "### [추가] 조건별 (전환 A)\n", strata_tables(a_rows), "",
                     "### [추가] 공식 라벨별 (전환 A, 전환 뒤 기준 회복률)\n",
                     label_table(a_rows, raw_labels), ""]

        if b_rows:
            b_md += [f"## {title}\n", baseline_table(b_rows, BASELINES), ""]

    # [추가] 여러 데이터셋에서 같은 뜻인 라벨을 합친 표 (평가 데이터셋만, train 은 뺌).
    # 영상 이름이 데이터셋끼리 겹칠 수 있어 "데이터셋/영상" 으로 구분한다.
    pooled = []
    for r in rows:
        if r["switch_set"] != "A" or r.get("part") is not None:
            continue
        merged = sorted({name for label in raw_labels(r)
                         if (name := extra_labels.common_name(r["dataset"], label))})
        if merged:
            pooled.append(dict(r, video=f"{r['dataset']}/{r['video']}", merged_labels=merged))
    extra_md += ["## [추가] 여러 데이터셋 합친 라벨 (전환 A, 전환 뒤 기준 회복률)\n",
                 "합치는 규칙은 benchmark/scoring/extra_labels.py 의 SAME_AS.\n",
                 label_table(pooled, lambda r: r["merged_labels"]), ""]

    out_dir = Path(settings.OUTPUT_ROOT) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in (("main.md", main_md), ("extra.md", extra_md), ("switch_b.md", b_md)):
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"저장: {out_dir / name}")


if __name__ == "__main__":
    main()
