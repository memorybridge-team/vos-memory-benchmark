"""SQLite 완료 키/현재 평가 필터와 과거 JSONL·분석 내보내기 도구.

새 추론 결과와 Native는 store.py에 저장한다. JSONL 파일은 재개에서 읽지 않는다.
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
    from evaluation.store import done_keys as database_done_keys
    return database_done_keys(dataset, videos=videos, run_ids=run_ids, seed=seed)


def current_rows(rows: list[dict], *, configuration=None, revisions=None) -> list[dict]:
    """현재 평가 버전, 전환 시점, 방법 실행 정의에 맞는 결과만 고른다."""
    return list(iter_current_rows(rows, configuration=configuration, revisions=revisions))


def iter_current_rows(rows, *, configuration=None, revisions=None):
    from evaluation.methods import METHODS

    revisions = {m.name: m.revision for m in METHODS} if revisions is None else revisions
    evaluation_revision = settings.EVALUATION_REVISION if configuration is None else configuration['evaluation_revision']
    fractions = settings.SWITCH_FRACTIONS if configuration is None else configuration['switch_fractions']
    minimum = settings.MIN_PRE_SWITCH_FRAMES if configuration is None else configuration['min_pre_switch_frames']
    from evaluation.switches import switch_points
    def eligible(row):
        return (row['switch_name'], row['switch_frame']) in {
            (s['name'], s['frame']) for s in switch_points(row['start'], row['end'],
                                                        fractions=fractions, min_pre_frames=minimum)}
    switches = {str(round(f * 100)) for f in fractions}
    return (r for r in rows
            if r.get("evaluation_revision") == evaluation_revision
            and type(r.get("run_id")) is int and r["run_id"] > 0
            and type(r.get("seed")) is int
            and r.get("native_reference_id")
            and r["switch_name"] in switches
            and r["baseline"] in revisions
            and r.get("baseline_revision", 1) == revisions[r["baseline"]]
            and eligible(r))


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
