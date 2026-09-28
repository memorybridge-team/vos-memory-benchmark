"""[4] MOSEv2 valid: 서버에서 받은 J&F → 결과 줄 + 회복률 (영상 전체 기준만).

    python scripts/4_mosev2_scores.py --csv mosev2_scores.csv

CSV (첫 줄은 머리글, j·f 는 없어도 됨):
    submission,video,jf,j,f
    source_only,00a1b2c3,0.712,0.690,0.734
    direct_state_copy__A50,00a1b2c3,0.705,0.681,0.729
    ...
  submission = 3_evaluate.py 가 만든 zip 이름 (확장자 빼고)
  video      = 영상 이름. 서버가 영상별 점수를 안 주면 비워 둔다 → 전체 점수끼리 나눔
               (이 경우 "영상마다 비율 평균"이 아니므로 표에 영상 수 1 로 나온다)
  점수는 0~1, 0~100 둘 다 받는다.

결과: outputs/mosev2/server_rows.jsonl (5_make_tables.py 가 같이 읽음)
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import records  # noqa: E402
from benchmark.methods import BY_NAME, METHODS  # noqa: E402
from benchmark.scoring import mosev2_server, retention  # noqa: E402

ALL_VIDEOS = "ALL"


def _score(text):
    if text is None or str(text).strip() == "":
        return None
    value = float(text)
    return value / 100 if value > 1 else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.csv, encoding="utf-8-sig") as f:
        for line in csv.DictReader(f):
            for method, switch_name in mosev2_server.parse_submission(line["submission"].strip()):
                rows.append({
                    "dataset": "mosev2_valid", "part": None,
                    "video": (line.get("video") or "").strip() or ALL_VIDEOS, "object": "all",
                    "switch_set": "A", "switch_name": switch_name,
                    "method": method, "role": BY_NAME[method].role,
                    "jf_whole": _score(line.get("jf")), "j_whole": _score(line.get("j")),
                    "f_whole": _score(line.get("f")), "source": "server",
                })

    out = mosev2_server.mosev2_root() / "server_rows.jsonl"
    out.unlink(missing_ok=True)
    records.append_rows(out, rows)
    print(f"저장: {out} ({len(rows)}줄)\n")

    replay_rows = [r for r in rows if r["method"] == "full_replay"]
    print("회복률 (영상 전체 기준, 영상마다 비율 → 평균)")
    for m in METHODS:
        m_rows = [r for r in rows if r["method"] == m.name]
        if not m_rows:
            continue
        ratios, dropped = retention.retention_by_video(m_rows, replay_rows, "jf_whole")
        text = f"{sum(ratios.values()) / len(ratios):.1f}" if ratios else "-"
        print(f"  {m.label:26s} {text}  (영상 {len(ratios)}개"
              + (f", Full Replay 0점이라 뺀 영상 {dropped}개" if dropped else "") + ")")


if __name__ == "__main__":
    main()
