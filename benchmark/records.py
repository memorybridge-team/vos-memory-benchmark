"""결과 한 줄씩 저장 (JSON Lines), 중단 후 이어하기.

줄 하나 = (영상, 객체, 전환 시점, 비교군) 하나. 파일: outputs/records/<데이터셋>.jsonl
객체 하나의 줄들은 다 만든 뒤 한꺼번에 쓴다 → 중간에 끊기면 그 객체만 다시 돌리면 된다.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

import settings


def records_path(dataset: str) -> Path:
    return Path(settings.OUTPUT_ROOT) / "records" / f"{dataset}.jsonl"


def _clean(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    return value


def append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({k: _clean(v) for k, v in row.items()}, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_rows(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_objects(path: Path) -> set[tuple[str, int]]:
    """이미 끝난 (영상, 객체)."""
    return {(r["video"], r["object"]) for r in read_rows(path)}
