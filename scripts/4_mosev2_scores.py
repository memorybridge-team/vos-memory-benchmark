"""[4] MOSEv2 valid: 서버에서 받은 J&F·J → 결과 줄 + 회복률 (영상 전체 기준만).

    python scripts/4_mosev2_scores.py --csv mosev2_scores.csv

CSV (첫 줄은 머리글, j 는 없어도 됨):
    submission,video,jf,j
    source_only,00a1b2c3,0.712,0.690
    direct_state_copy__50,00a1b2c3,0.705,0.681
    ...
  submission = 3_evaluate.py 가 만든 zip 이름 (확장자 빼고)
  video      = 영상 이름. 서버가 영상별 점수를 안 주면 비워 둔다 → 전체 점수끼리 나눔
               (이 경우 "영상마다 비율 평균"이 아니므로 표에 영상 수 1 로 나온다)
  점수는 0~1, 0~100 둘 다 받는다.

결과: outputs/mosev2/server_rows.jsonl (5_make_tables.py 가 같이 읽음)
  점수는 결과 줄의 j, jf 열에 들어간다 (다른 데이터셋은 전환 뒤 점수, MOSEv2 는 영상 전체 점수).
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import records  # noqa: E402
from baseline import BY_NAME, BASELINES  # noqa: E402
from evaluation.scoring import main_metrics, mosev2_server  # noqa: E402

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
            for baseline, switch_name in mosev2_server.parse_submission(line["submission"].strip()):
                rows.append({
                    "dataset": "mosev2_valid", "part": None,
                    "video": (line.get("video") or "").strip() or ALL_VIDEOS, "object": "all",
                    "switch_name": switch_name,
                    "baseline": baseline, "role": BY_NAME[baseline].role,
                    "jf": _score(line.get("jf")), "j": _score(line.get("j")), "source": "server",
                })

    out = mosev2_server.mosev2_root() / "server_rows.jsonl"
    out.unlink(missing_ok=True)
    records.append_rows(out, rows)
    print(f"저장: {out} ({len(rows)}줄)\n")

    replay_rows = [r for r in rows if r["baseline"] == "full_replay"]
    print("회복률 J&F (영상 전체 기준, 영상마다 비율 → 평균)")
    for m in BASELINES:
        m_rows = [r for r in rows if r["baseline"] == m.name]
        if m_rows:
            value = main_metrics.retention(m_rows, replay_rows, "jf")
            print(f"  {m.label:26s} {'-' if value is None else f'{value:.1f}'}")


if __name__ == "__main__":
    main()
