"""[3] 결과 줄 → 표.

    python scripts/3_make_tables.py

읽는 것: outputs/records/*.jsonl, outputs/lists/*.json (공식 라벨)
만드는 것 (outputs/tables/):
  main.md   주 표: 확정 비교군 9개 × 전환 25/50/75% — J·J&F·회복률·격차 회복률·시간
            (evaluation/tables/main_tables.py)
  extra.md  추가 표: 비용 세부, 출력 일치도, 진단 비교군, 실패 분석, 공식 라벨별
            (evaluation/tables/extra_tables.py)
점수는 100점 만점, 영상 평균 (VOS 벤치마크 관례대로 숫자 하나).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from evaluation import records  # noqa: E402
from evaluation.tables import extra_tables, main_tables  # noqa: E402


def load_rows() -> list[dict]:
    rows = []
    for path in sorted((Path(settings.OUTPUT_ROOT) / "records").glob("*.jsonl")):
        rows += records.read_rows(path)
    return records.unique_rows(rows)


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
    groups = defaultdict(list)
    for r in load_rows():
        groups[r["dataset"]].append(r)
    groups = dict(sorted(groups.items()))

    out_dir = Path(settings.OUTPUT_ROOT) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"main.md": main_tables.build(groups),
              "extra.md": extra_tables.build(groups, load_object_labels())}
    for name, lines in tables.items():
        (out_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"저장: {out_dir / name}")


if __name__ == "__main__":
    main()
