"""SQLite 원본: 조건 메타데이터, 프레임 점수, 기억 R², Native 점수와 tensor BLOB.

객체 결과 한 묶음은 한 transaction으로 저장한다. GPU 측정 구간 밖에서 호출한다.
CSV/JSONL/PDF는 공유용 export이며 원본은 OUTPUT_ROOT/benchmark.sqlite3다.
"""

from contextlib import contextmanager
import io
import json
import sqlite3
from pathlib import Path

import settings

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, revision INTEGER NOT NULL,
    baseline_revision INTEGER NOT NULL, run_id INTEGER NOT NULL, seed INTEGER NOT NULL,
    video TEXT NOT NULL, object_id INTEGER NOT NULL, start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL, switch_name TEXT NOT NULL, switch_frame INTEGER NOT NULL,
    baseline TEXT NOT NULL, native_reference_id TEXT, payload TEXT NOT NULL,
    UNIQUE(dataset,revision,baseline_revision,run_id,seed,video,object_id,start_frame,end_frame,
           switch_name,switch_frame,baseline));
CREATE INDEX IF NOT EXISTS evaluations_lookup ON evaluations(dataset,revision,seed,run_id,video);
CREATE TABLE IF NOT EXISTS frame_scores (
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    phase TEXT NOT NULL CHECK(phase IN ('pre','post')), frame INTEGER NOT NULL,
    j REAL, f REAL, jf REAL, native_run_j REAL, native_run_jf REAL,
    native_j REAL, native_jf REAL, recovery_j REAL, recovery_jf REAL, payload TEXT NOT NULL,
    PRIMARY KEY(evaluation_id,phase,frame));
CREATE TABLE IF NOT EXISTS restoration_scores (
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    frame INTEGER NOT NULL, spatial_r2 REAL, pointer_r2 REAL,
    spatial_sse REAL, spatial_sst REAL, pointer_sse REAL, pointer_sst REAL,
    payload TEXT NOT NULL, PRIMARY KEY(evaluation_id,frame));
CREATE TABLE IF NOT EXISTS native_references (
    reference_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, revision INTEGER NOT NULL,
    run_id INTEGER NOT NULL, seed INTEGER NOT NULL, video TEXT NOT NULL, object_id INTEGER NOT NULL,
    start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL, schedule TEXT NOT NULL, payload TEXT NOT NULL,
    UNIQUE(dataset,revision,run_id,seed,video,object_id,start_frame,end_frame,schedule));
CREATE INDEX IF NOT EXISTS native_lookup ON native_references(dataset,revision,seed,run_id,video);
CREATE TABLE IF NOT EXISTS native_scores (
    reference_id TEXT NOT NULL REFERENCES native_references(reference_id) ON DELETE CASCADE,
    frame INTEGER NOT NULL, j REAL, f REAL, gt_visible INTEGER NOT NULL,
    payload TEXT NOT NULL, PRIMARY KEY(reference_id,frame));
CREATE TABLE IF NOT EXISTS native_memory (
    reference_id TEXT PRIMARY KEY, tensor_blob BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    namespace TEXT NOT NULL, name TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(namespace,name));
CREATE TABLE IF NOT EXISTS analysis_rows (
    tag TEXT NOT NULL, row_index INTEGER NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(tag,row_index));
CREATE TABLE IF NOT EXISTS legacy_imports (
    path TEXT PRIMARY KEY, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL);
"""


def database_path():
    configured = settings.DATABASE_PATH
    return Path(configured) if configured else Path(settings.OUTPUT_ROOT) / 'benchmark.sqlite3'


def _json(value):
    def scalar(x):
        if hasattr(x, 'item'):
            return x.item()
        raise TypeError(f'JSON에 저장할 수 없는 값: {type(x)}')
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False, default=scalar)


@contextmanager
def connection():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=settings.SQLITE_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        if version == 0:
            conn.executescript(SCHEMA)
            conn.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
        elif version != SCHEMA_VERSION:
            raise ValueError(f'SQLite schema 버전이 다릅니다: {version}, {path}')
        with conn:
            yield conn
    finally:
        conn.close()


FRAME_VALUES = ('j','f','jf','native_run_j','native_run_jf','native_j','native_jf','recovery_j','recovery_jf')
POINT_FIELDS = ('frame_scores','pre_switch_frame_scores','restoration_frame_scores')


def _save_evaluation(conn, row):
    metadata = {k:v for k,v in row.items() if k not in POINT_FIELDS}
    key = (row['dataset'],row.get('evaluation_revision',-1),row.get('baseline_revision',1),
           row.get('run_id',-1),row.get('seed',-1),row['video'],row['object'],
           row['start'],row['end'],row['switch_name'],row['switch_frame'],row['baseline'])
    existing = conn.execute('SELECT id,native_reference_id FROM evaluations WHERE '
                            'dataset=? AND revision=? AND baseline_revision=? AND run_id=? AND seed=? '
                            'AND video=? AND object_id=? AND start_frame=? AND end_frame=? '
                            'AND switch_name=? AND switch_frame=? AND baseline=?',key).fetchone()
    if existing is not None:
        if existing['native_reference_id'] != row.get('native_reference_id'):
            raise ValueError('같은 평가 조건의 Native 기준이 서로 다릅니다.')
        return False
    cursor = conn.execute('INSERT INTO evaluations(dataset,revision,baseline_revision,run_id,seed,video,object_id,'
                          'start_frame,end_frame,switch_name,switch_frame,baseline,native_reference_id,payload) '
                          'VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',key+(row.get('native_reference_id'),_json(metadata)))
    identity = cursor.lastrowid
    for phase,field in (('pre','pre_switch_frame_scores'),('post','frame_scores')):
        conn.executemany('INSERT INTO frame_scores VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                         ((identity,phase,p['frame'],*(p.get(k) for k in FRAME_VALUES),_json(p))
                          for p in row.get(field,[])))
    conn.executemany('INSERT INTO restoration_scores VALUES(?,?,?,?,?,?,?,?,?)',
                     ((identity,p['frame'],p.get('maskmem_features',{}).get('r2'),p.get('obj_ptr',{}).get('r2'),
                       p.get('maskmem_features',{}).get('sse'),p.get('maskmem_features',{}).get('sst'),
                       p.get('obj_ptr',{}).get('sse'),p.get('obj_ptr',{}).get('sst'),_json(p))
                      for p in row.get('restoration_frame_scores',[])))
    return True


def save_evaluations(rows):
    if not rows:
        return 0
    with connection() as conn:
        return sum(_save_evaluation(conn,row) for row in rows)


def _where(dataset=None, seed=None, runs=None, run_ids=None, videos=None, revision=None):
    clauses, args = [], []
    for field,value in (('dataset',dataset),('seed',seed),('revision',revision)):
        if value is not None:
            clauses.append(f'{field}=?');args.append(value)
    if runs is not None:
        clauses.append('run_id<=?');args.append(runs)
    for field,values in (('run_id',run_ids),('video',videos)):
        if values is not None:
            values = list(values)
            if not values:
                clauses.append('0')
            else:
                clauses.append(f"{field} IN ({','.join('?' for _ in values)})");args.extend(values)
    return (' WHERE '+' AND '.join(clauses) if clauses else ''), args


def iter_evaluations(dataset=None, *, seed=None, runs=None, run_ids=None, videos=None, revision=None):
    if not database_path().exists():
        return
    where,args = _where(dataset,seed,runs,run_ids,videos,revision)
    with connection() as conn:
        for entry in conn.execute('SELECT id,payload FROM evaluations'+where+' ORDER BY id',args):
            row = json.loads(entry['payload'])
            for phase,field in (('pre','pre_switch_frame_scores'),('post','frame_scores')):
                row[field] = [json.loads(p[0]) for p in conn.execute(
                    'SELECT payload FROM frame_scores WHERE evaluation_id=? AND phase=? ORDER BY frame',
                    (entry['id'],phase))]
            row['restoration_frame_scores'] = [json.loads(p[0]) for p in conn.execute(
                'SELECT payload FROM restoration_scores WHERE evaluation_id=? ORDER BY frame',(entry['id'],))]
            yield row


def done_keys(dataset, *, seed=None, run_ids=None, videos=None):
    if not database_path().exists():
        return set()
    from evaluation.records import iter_current_rows, row_key
    where,args = _where(dataset,seed,None,run_ids,videos,settings.EVALUATION_REVISION)
    with connection() as conn:
        # 프레임별 점수/BLOB을 읽지 않고 조건 메타데이터만 검사한다.
        metadata = (json.loads(r[0]) for r in conn.execute('SELECT payload FROM evaluations'+where,args))
        return {row_key(r) for r in iter_current_rows(metadata)}


def _save_reference(conn, ref):
    metadata = {k:v for k,v in ref.items() if k != 'scores'}
    schedule = _json(ref['switches'])
    key = (ref['dataset'],ref.get('evaluation_revision',-1),ref['run_id'],ref['seed'],ref['video'],
           ref['object'],ref['start'],ref['end'],schedule)
    previous = conn.execute('SELECT reference_id FROM native_references WHERE dataset=? AND revision=? '
                            'AND run_id=? AND seed=? AND video=? AND object_id=? AND start_frame=? '
                            'AND end_frame=? AND schedule=?',key).fetchone()
    if previous is not None:
        if previous[0] != ref['native_reference_id']:
            raise ValueError('동일 조건의 Native 기준이 둘 이상입니다.')
        return
    conn.execute('INSERT INTO native_references VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                 (ref['native_reference_id'],*key,_json(metadata)))
    conn.executemany('INSERT INTO native_scores VALUES(?,?,?,?,?,?)',
                     ((ref['native_reference_id'],p['frame'],p['j'],p['f'],int(p['gt_visible']),_json(p))
                      for p in ref['scores']))


def serialize_memory(ref, snapshots):
    import torch
    from evaluation.scoring import restoration
    buffer = io.BytesIO()
    torch.save({'native_reference_id':ref['native_reference_id'],
                'native_memory_revision':restoration.REVISION,'snapshots':snapshots},buffer)
    return buffer.getvalue()


def save_native(ref, snapshots):
    # scalar 기준 + 모든 전환의 tensor를 하나의 transaction으로 확정한다.
    blob = serialize_memory(ref,snapshots)
    with connection() as conn:
        _save_reference(conn,ref)
        conn.execute('INSERT OR IGNORE INTO native_memory VALUES(?,?)',(ref['native_reference_id'],blob))


def iter_references(dataset, *, seed=None, run_ids=None, videos=None):
    if not database_path().exists():
        return
    where,args = _where(dataset,seed,None,run_ids,videos,settings.EVALUATION_REVISION)
    with connection() as conn:
        for entry in conn.execute('SELECT reference_id,payload FROM native_references'+where+' ORDER BY rowid',args):
            ref = json.loads(entry['payload'])
            ref['scores'] = [json.loads(p[0]) for p in conn.execute(
                'SELECT payload FROM native_scores WHERE reference_id=? ORDER BY frame',(entry['reference_id'],))]
            yield ref


def memory_blob(identity):
    if not database_path().exists():
        return None
    with connection() as conn:
        row = conn.execute('SELECT tensor_blob FROM native_memory WHERE reference_id=?',(identity,)).fetchone()
        return row[0] if row is not None else None


def put_document(namespace,name,value):
    with connection() as conn:
        conn.execute('INSERT INTO documents VALUES(?,?,?) ON CONFLICT(namespace,name) DO UPDATE SET payload=excluded.payload',
                     (namespace,name,_json(value)))


def save_analysis(tag,rows):
    with connection() as conn:
        conn.execute('DELETE FROM analysis_rows WHERE tag=?',(tag,))
        conn.executemany('INSERT INTO analysis_rows VALUES(?,?,?)',((tag,i,_json(r)) for i,r in enumerate(rows)))


def import_legacy(dataset=None):
    """이전 JSONL/.pt를 DB에 추가한다. 원본 파일은 수정/삭제하지 않는다. 재실행은 멱등이다."""
    from evaluation.records import iter_rows
    report = {'evaluations':0,'references':0,'memory_blobs':0}
    root = Path(settings.OUTPUT_ROOT)
    pattern = f'{dataset}*.jsonl' if dataset else '*.jsonl'
    for folder in ('native','records'):
        for path in sorted((root/folder).glob(pattern)):
            stat = path.stat()
            with connection() as conn:
                old = conn.execute('SELECT size,mtime_ns FROM legacy_imports WHERE path=?',(str(path.resolve()),)).fetchone()
                if old is not None and tuple(old) == (stat.st_size,stat.st_mtime_ns):
                    continue
                for row in iter_rows(path):
                    if folder == 'records':
                        report['evaluations'] += _save_evaluation(conn,row)
                    else:
                        _save_reference(conn,row);report['references'] += 1
                        from evaluation.native import memory_path
                        memory = memory_path(row)
                        if memory.exists():
                            inserted = conn.execute('INSERT OR IGNORE INTO native_memory VALUES(?,?)',
                                                    (row['native_reference_id'],memory.read_bytes()))
                            report['memory_blobs'] += inserted.rowcount
                conn.execute('INSERT INTO legacy_imports VALUES(?,?,?) ON CONFLICT(path) DO UPDATE SET '
                             'size=excluded.size,mtime_ns=excluded.mtime_ns',
                             (str(path.resolve()),stat.st_size,stat.st_mtime_ns))
    return report
