"""조건 키·현재 평가 모집단 검증과 이전 JSONL import/export 보조 함수.

평가 원본은 store.py의 SQLite에 저장한다. JSONL은 이전 파일 읽기와 분석 export에 쓴다.
객체 전체 결과는 하나의 DB transaction으로 확정하므로 중단 시 일부만 완료되지 않는다.
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
    from evaluation import store
    store.import_legacy(dataset)
    return store.done_keys(dataset, videos=videos, run_ids=run_ids, seed=seed)


def current_rows(rows: list[dict]) -> list[dict]:
    """현재 평가 버전, 전환 시점, 방법 실행 정의에 맞는 결과만 고른다."""
    return list(iter_current_rows(rows))


def iter_current_rows(rows):
    from evaluation.methods import METHODS

    from evaluation.switches import object_exclusion, switch_points
    revisions = {m.name: m.revision for m in METHODS}
    switches = {str(round(f * 100)) for f in settings.SWITCH_FRACTIONS}
    for r in rows:
        if not (r.get('evaluation_revision') == settings.EVALUATION_REVISION
                and type(r.get('run_id')) is int and r['run_id'] > 0
                and type(r.get('seed')) is int and r.get('native_reference_id')
                and r.get('switch_name') in switches and r.get('baseline') in revisions
                and r.get('baseline_revision',1) == revisions[r['baseline']]
                and type(r.get('start')) is int and type(r.get('end')) is int):
            continue
        obj = {'start':r['start'],'end':r['end'],'switches':switch_points(r['start'],r['end'])}
        if object_exclusion(obj) is not None:
            continue
        if {'name':r['switch_name'],'frame':r['switch_frame']} not in obj['switches']:
            continue
        yield r


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
