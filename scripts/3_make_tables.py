"""[3] 현재 평가 결과 → 전환별 요약, 난이도 성능표와 전환 후 경과 프레임별 CSV.

    python scripts/3_make_tables.py

outputs/tables/main.md: 50/75%별 J·J&F, 기존 회복률, 전환시간·GPU 메모리, 실패비율
outputs/tables/extra.md: 공식 라벨과 공통 난이도 유형별 성능
outputs/tables/temporal.csv: 전환 후 경과 프레임별 J·J&F(100점 만점)와 집계 표본 수
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from evaluation import records  # noqa: E402
from evaluation.tables import extra_tables, main_tables, temporal  # noqa: E402


def load_rows() -> list[dict]:
    rows = []
    for path in sorted((Path(settings.OUTPUT_ROOT) / "records").glob("*.jsonl")):
        rows += records.read_rows(path)
    return records.unique_rows(records.current_rows(rows))


def load_object_labels() -> dict:
    """(데이터셋, 영상, 객체) → 공식 라벨. 1_make_video_list.py 가 목록에 적어 둔 것."""
    labels = {}
    for path in sorted((Path(settings.OUTPUT_ROOT) / "lists").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data["videos"]:
            for obj in entry["objects"]:
                labels[(data["dataset"], entry["video"], obj["object"])] = obj.get("extra_labels", [])
    return labels


def main():
    rows = load_rows()
    groups = defaultdict(list)
    for r in rows:
        groups[r["dataset"]].append(r)
    groups = dict(sorted(groups.items()))

    out_dir = Path(settings.OUTPUT_ROOT) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"main.md": main_tables.build(groups),
              "extra.md": extra_tables.build(groups, load_object_labels())}
    for name, lines in tables.items():
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"저장: {out_dir / name}")

    curve_path = out_dir / "temporal.csv"
    with curve_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=temporal.CSV_FIELDS)
        writer.writeheader()
        writer.writerows(temporal.build(rows))
    print(f"저장: {curve_path}")


if __name__ == "__main__":
    main()
