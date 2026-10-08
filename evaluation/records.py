"""결과 한 줄씩 저장 (JSON Lines), 중단 후 이어하기.

줄 하나 = (반복 번호, seed, 영상, 객체, 전환 시점, 방법) 하나. 파일: outputs/records/<데이터셋>[.shard].jsonl
(예전에 따로 돌린 <데이터셋>.model.jsonl · .baselines.jsonl 도 같이 읽는다)
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


def write_rows(path: Path, rows: list[dict]) -> None:
    """재집계 결과를 원자적으로 교체한다. 원본 records/native에는 사용하지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps({k: _clean(v) for k, v in row.items()}, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def iter_rows(path: Path):
    """큰 프레임별 JSONL을 한 줄씩 읽어 메모리 사용량을 제한한다."""
    if not Path(path).exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_rows(path: Path) -> list[dict]:
    return list(iter_rows(path))


def row_key(row: dict) -> tuple:
    """이어하기 키에 반복 번호·seed를 포함한다."""
    return (row.get("run_id", 1), row.get("seed", settings.EVALUATION_SEED),
            row["video"], row["object"], row["switch_name"], row["switch_frame"], row["baseline"])


def done_keys(dataset: str, *, videos=None, run_ids=None, seed=None) -> set[tuple]:
    """현재 정의로 끝난 결과 줄. 일부 전환만 완료된 방법도 나머지는 이어서 계산한다."""
    folder = records_path(dataset).parent
    selected = None if videos is None else set(videos)
    runs = None if run_ids is None else set(run_ids)
    return {row_key(r) for path in folder.glob(f"{dataset}*.jsonl")
            for r in iter_current_rows(iter_rows(path))
            if (selected is None or r['video'] in selected)
            and (runs is None or r['run_id'] in runs)
            and (seed is None or r['seed'] == seed)}


def current_rows(rows: list[dict]) -> list[dict]:
    """현재 평가 버전, 전환 시점, 방법 실행 정의에 맞는 결과만 고른다."""
    return list(iter_current_rows(rows))


def iter_current_rows(rows):
    from evaluation.methods import METHODS

    revisions = {m.name: m.revision for m in METHODS}
    switches = {str(round(f * 100)) for f in settings.SWITCH_FRACTIONS}
    return (r for r in rows
            if r.get("evaluation_revision") == settings.EVALUATION_REVISION
            and type(r.get("run_id")) is int and r["run_id"] > 0
            and type(r.get("seed")) is int
            and r.get("native_reference_id")
            and r["switch_name"] in switches
            and r["baseline"] in revisions
            and r.get("baseline_revision", 1) == revisions[r["baseline"]])


def unique_rows(rows: list[dict]) -> list[dict]:
    """같은 (데이터셋, 반복, seed, 영상, 객체, 전환, 방법) 줄은 처음 것만.
    예전에 본 모델·비교군을 따로 돌린 결과에는 Full Replay · Source-only 줄이 두 번 있을 수 있다."""
    seen, out = set(), []
    for r in rows:
        key = (r["dataset"], *row_key(r))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out
