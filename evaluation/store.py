"""SQLite 원점수 저장. GPU 계산 밖에서 실행 단위 전체를 원자적으로 커밋한다.

JSONL은 과거 자료/분석 내보내기에만 사용한다. 새 평가의 재개 기준은 이 DB이다.
"""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
from itertools import chain
import json
from pathlib import Path
import sqlite3

import settings
from evaluation.scoring import restoration

SCHEMA_VERSION = 1
POINT_FIELDS = ('pre_switch_frame_scores', 'frame_scores', 'frame_times', 'restoration_frame_scores')
_TABLE_COLUMNS = {}


def database_path():
    return Path(settings.OUTPUT_ROOT) / 'benchmark.sqlite'


def dumps(value):
    from evaluation.records import _clean
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=_clean)


@lru_cache(maxsize=16)
def _file_hash(path, size, modified):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def experiment():
    """회차/방법 선택과 라벨 수정은 실행 정의를 바꾸지 않는다."""
    models = {}
    for name in (settings.SOURCE_MODEL, settings.TARGET_MODEL):
        spec = dict(settings.MODELS[name])
        path = Path(settings.SAM2_CHECKPOINT_DIR) / spec['checkpoint']
        if path.is_file():
            stat = path.stat()
            spec['checkpoint_sha256'] = _file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        models[name] = spec
    config = dict(evaluation_revision=settings.EVALUATION_REVISION,
                  runtime_revision=settings.EVALUATION_RUNTIME_REVISION,
                  native_memory_revision=restoration.REVISION,
                  source_model=settings.SOURCE_MODEL, target_model=settings.TARGET_MODEL, models=models,
                  translator_sha256=settings.TRANSLATOR_SHA256,
                  data_root=str(Path(settings.DATA_ROOT).resolve()), data_folders=settings.DATA_FOLDERS,
                  lvos_split=settings.LVOS_SPLIT_FOLDER, lvos_attributes=settings.LVOS_ATTRIBUTE_FILE,
                  switch_fractions=settings.SWITCH_FRACTIONS,
                  min_pre_switch_frames=settings.MIN_PRE_SWITCH_FRAMES,
                  memory_window=settings.MEMORY_WINDOW, replay_frames=settings.REPLAY_FRAMES,
                  device=settings.DEVICE, use_bf16=settings.USE_BF16,
                  offload_state_to_cpu=settings.OFFLOAD_STATE_TO_CPU,
                  boundary_threshold=settings.BOUNDARY_THRESHOLD, recall_j=settings.RECALL_J,
                  prefetch_frames=settings.PREFETCH_FRAMES)
    report = Path(settings.OUTPUT_ROOT) / 'checks' / 'sam2.json'
    if report.is_file():
        # require_passed가 현재 설치와의 일치를 검사한다. 가짜 테스트에는 이 기록이 없다.
        try:
            config['checked_runtime'] = json.loads(report.read_text())['runtime']
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f'{report}를 읽을 수 없습니다. 0_check_sam2.py를 다시 실행하세요.') from exc
    encoded = dumps(config)
    return hashlib.sha256(encoded.encode()).hexdigest(), encoded


@contextmanager
def connect():
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=settings.SQLITE_BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        mode = settings.SQLITE_JOURNAL_MODE.upper()
        if mode not in ('DELETE', 'WAL'):
            raise ValueError('SQLITE_JOURNAL_MODE는 DELETE 또는 WAL이어야 합니다.')
        if conn.execute('PRAGMA journal_mode').fetchone()[0].upper() != mode:
            conn.execute(f'PRAGMA journal_mode={mode}')
        version = conn.execute('PRAGMA user_version').fetchone()[0]
        if version == 0:
            conn.executescript('BEGIN IMMEDIATE;\n' + Path(__file__).with_name('schema.sql').read_text() + '\nCOMMIT;')
        elif version != SCHEMA_VERSION:
            raise ValueError(f'지원하지 않는 SQLite 스키마 버전: {version}')
        with conn:
            yield conn
    finally:
        conn.close()


def _fields(conn, table, row):
    if table not in _TABLE_COLUMNS:
        _TABLE_COLUMNS[table] = tuple(p['name'] for p in conn.execute(f'PRAGMA table_info({table})'))
    return [name for name in _TABLE_COLUMNS[table] if name in row]


def _insert(conn, table, row):
    # table/column 이름은 코드에서 고정한다. 모든 값은 파라미터로 전달한다.
    fields = _fields(conn, table, row)
    return conn.execute(f'INSERT INTO {table} ({",".join(fields)}) VALUES ({",".join("?" for _ in fields)})',
                        [row[f] for f in fields])


def _insert_many(conn, table, rows):
    it = iter(rows)
    first = next(it, None)
    if first is None:
        return
    fields = _fields(conn, table, first)
    conn.executemany(f'INSERT INTO {table} ({",".join(fields)}) VALUES ({",".join("?" for _ in fields)})',
                     ([row[f] for f in fields] for row in chain((first,), it)))


def _register(conn, identity, configuration):
    conn.execute('INSERT OR IGNORE INTO experiments VALUES (?,?)', (identity, configuration))


def _coverage(points, start, end):
    frames = [p['frame'] for p in points]
    if len(frames) != end - start + 1 or set(frames) != set(range(start, end + 1)):
        raise ValueError(f'전체 프레임 원점수가 누락/중복되었습니다: {start}~{end}')


def save_native(ref):
    from evaluation.native import memory_path
    _coverage(ref['scores'], ref['start'], ref['end'])
    _coverage(ref['frame_times'], ref['start'], ref['end'])
    if not memory_path(ref).is_file():
        raise ValueError('Native 기억 파일을 먼저 저장해야 합니다.')
    identity, config = experiment()
    metadata = {k: v for k, v in ref.items() if k not in ('scores', 'frame_times', 'switches', 'costs')}
    with connect() as conn:
        _register(conn, identity, config)
        _insert(conn, 'native_runs', dict(metadata, experiment_id=identity, object_id=ref['object'],
                start_frame=ref['start'], end_frame=ref['end'], memory_path=str(memory_path(ref)),
                switch_signature=dumps(ref['switches']), metadata=dumps(metadata)))
        for sw in ref['switches']:
            _insert(conn, 'native_switches', dict(native_reference_id=ref['native_reference_id'],
                    switch_name=sw['name'], switch_frame=sw['frame'], **ref['costs'][str(sw['frame'])]))
        _insert_many(conn, 'native_frame_scores',
                     (dict(p, native_reference_id=ref['native_reference_id']) for p in ref['scores']))
        _insert_many(conn, 'native_frame_times',
                     (dict(p, native_reference_id=ref['native_reference_id']) for p in ref['frame_times']))


def load_native(dataset, *, videos=None, run_ids=None, seed=None):
    if not database_path().exists():
        return []
    identity, _ = experiment()
    out = []
    with connect() as conn:
        for raw in conn.execute('SELECT * FROM native_runs WHERE experiment_id=? AND dataset=?', (identity, dataset)):
            if (videos is not None and raw['video'] not in videos or
                    run_ids is not None and raw['run_id'] not in run_ids or
                    seed is not None and raw['seed'] != seed):
                continue
            ref = json.loads(raw['metadata'])
            rid = ref['native_reference_id']
            ref['switches'], ref['costs'] = [], {}
            for sw in conn.execute('SELECT * FROM native_switches WHERE native_reference_id=? ORDER BY switch_frame', (rid,)):
                ref['switches'].append({'name': sw['switch_name'], 'frame': sw['switch_frame']})
                ref['costs'][str(sw['switch_frame'])] = {k: sw[k] for k in ('switch_seconds', 'switch_gpu_mb')}
            ref['scores'] = [{k: p[k] for k in ('frame', 'has_gt', 'gt_visible', 'j', 'f', 'jf')}
                             for p in conn.execute('SELECT * FROM native_frame_scores WHERE native_reference_id=? ORDER BY frame', (rid,))]
            ref['frame_times'] = [{k: p[k] for k in ('frame', 'seconds', 'gpu_peak_mb')}
                                  for p in conn.execute('SELECT * FROM native_frame_times WHERE native_reference_id=? ORDER BY frame', (rid,))]
            _coverage(ref['scores'], ref['start'], ref['end'])
            _coverage(ref['frame_times'], ref['start'], ref['end'])
            # DB는 완료인데 외부 기억 파일이 없으면 새 기준으로 몰래 교체하지 않는다.
            from evaluation.native import memory_path
            if not Path(raw['memory_path']).is_file() or Path(raw['memory_path']).resolve() != memory_path(ref).resolve():
                raise ValueError(f'Native 기억 기준 파일이 없습니다/경로가 다릅니다: {raw["memory_path"]}')
            out.append(ref)
    return out


def save_results(rows):
    if not rows:
        return
    identity, config = experiment()
    with connect() as conn:
        _register(conn, identity, config)
        for row in rows:
            points = row['pre_switch_frame_scores'] + row['frame_scores']
            _coverage(points, row['start'], row['end'])
            _coverage(row['frame_times'], row['start'], row['end'])
            reference = conn.execute('SELECT * FROM native_runs WHERE native_reference_id=?',
                                     (row['native_reference_id'],)).fetchone()
            if reference is None or any(reference[k] != v for k, v in {
                'experiment_id': identity, 'dataset': row['dataset'], 'video': row['video'],
                'object_id': row['object'], 'start_frame': row['start'], 'end_frame': row['end'],
                'run_id': row['run_id'], 'seed': row['seed']}.items()):
                raise ValueError('결과와 Native의 실행 조건이 다릅니다.')
            if not conn.execute('SELECT 1 FROM native_switches WHERE native_reference_id=? AND switch_name=? AND switch_frame=?',
                                (row['native_reference_id'], row['switch_name'], row['switch_frame'])).fetchone():
                raise ValueError('Native에 대응하는 전환 기준이 없습니다.')
            metadata = {k: v for k, v in row.items() if k not in (*POINT_FIELDS, 'result_id', 'experiment_id')}
            result = dict(metadata, experiment_id=identity, object_id=row['object'],
                          start_frame=row['start'], end_frame=row['end'],
                          extra_labels=dumps(row['extra_labels']), metadata=dumps(metadata))
            rid = _insert(conn, 'results', result).lastrowid
            for phase, field in (('pre', 'pre_switch_frame_scores'), ('post', 'frame_scores')):
                for p in row[field]:
                    if (p['frame'] <= row['switch_frame']) != (phase == 'pre'):
                        raise ValueError('전환 전후 프레임 구간이 다릅니다.')
                _insert_many(conn, 'frame_scores', (dict(p, result_id=rid, phase=phase) for p in row[field]))
            _insert_many(conn, 'result_frame_times', (dict(p, result_id=rid) for p in row['frame_times']))
            for p in row['restoration_frame_scores']:
                _insert(conn, 'restoration_frames', dict(p, result_id=rid))
                for field in restoration.FIELDS:
                    part = dict(p[field], result_id=rid, frame=p['frame'], field=field)
                    for shape in ('native_shape', 'target_shape'):
                        part[shape] = dumps(part[shape]) if part[shape] is not None else None
                    _insert(conn, 'restoration_fields', part)


def list_experiments():
    """파일/체크포인트를 검사하지 않고 DB의 실험 기록만 읽는다."""
    if not database_path().exists():
        return []
    out = []
    with connect() as conn:
        for raw in conn.execute('SELECT * FROM experiments ORDER BY experiment_id'):
            identity = raw['experiment_id']
            results = conn.execute('SELECT COUNT(*) AS n FROM results WHERE experiment_id=?', (identity,)).fetchone()['n']
            natives = conn.execute('SELECT COUNT(*) AS n FROM native_runs WHERE experiment_id=?', (identity,)).fetchone()['n']
            revisions = {r['baseline']: r['revision'] for r in conn.execute(
                'SELECT baseline, MAX(baseline_revision) AS revision FROM results WHERE experiment_id=? GROUP BY baseline',
                (identity,))}
            samples = conn.execute('SELECT DISTINCT dataset, seed, run_id FROM results WHERE experiment_id=? ORDER BY dataset, seed, run_id',
                                   (identity,)).fetchall()
            out.append(dict(experiment_id=identity, configuration=json.loads(raw['configuration']),
                            result_count=results, native_count=natives, baseline_revisions=revisions,
                            datasets=sorted({r['dataset'] for r in samples}),
                            seeds=sorted({r['seed'] for r in samples}), run_ids=sorted({r['run_id'] for r in samples})))
    return out


def select_experiment(experiment_id=None):
    experiments = list_experiments()
    if experiment_id is not None:
        selected = next((e for e in experiments if e['experiment_id'] == experiment_id), None)
        if selected is None:
            raise ValueError(f'DB에 없는 experiment_id: {experiment_id}. --list-experiments로 확인하세요.')
        return selected
    if len(experiments) != 1:
        ids = ', '.join(e['experiment_id'] for e in experiments)
        raise ValueError(f'DB 실험이 {len(experiments)}개입니다. --list-experiments로 확인하고 --experiment-id를 지정하세요. {ids}')
    return experiments[0]


@contextmanager
def analysis_settings(selected):
    """기존 계산식은 유지하고 선택한 실험의 평가 설정만 적용한 뒤 복구한다."""
    config = selected['configuration']
    values = {'SWITCH_FRACTIONS': tuple(config['switch_fractions']),
              'MIN_PRE_SWITCH_FRAMES': config['min_pre_switch_frames'],
              'EVALUATION_REVISION': config['evaluation_revision'], 'RECALL_J': config['recall_j'],
              # 스키마 1 실험에는 주 지표 목록이 없다. 당시 프로토콜의 기본값을 쓴다.
              'J_MAIN_DATASETS': tuple(config.get('j_main_datasets', ('vost_val', 'm3vos')))}
    previous = {k: getattr(settings, k) for k in values}
    try:
        for key, value in values.items():
            setattr(settings, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(settings, key, value)


def analysis_video_lists(experiment_id, *, seed, runs):
    """DB에 저장된 Native 객체 구간을 이용한다. 로컬 영상 목록은 읽지 않는다."""
    lists = {}
    with connect() as conn:
        query = '''SELECT n.dataset, n.video, n.object_id, n.start_frame, n.end_frame, s.switch_name, s.switch_frame
                   FROM native_runs n JOIN native_switches s USING(native_reference_id)
                   WHERE n.experiment_id=? AND n.seed=? AND n.run_id<=?'''
        for row in conn.execute(query, (experiment_id, seed, runs)):
            plan = lists.setdefault(row['dataset'], {'window_basis': 'stored_native_object_ranges', 'videos': {}})
            video = plan['videos'].setdefault(row['video'], {'video': row['video'], 'objects': {}})
            key = (row['object_id'], row['start_frame'], row['end_frame'])
            obj = video['objects'].setdefault(key, {'object': key[0], 'start': key[1], 'end': key[2], 'switches': {}})
            obj['switches'][row['switch_name']] = {'name': row['switch_name'], 'frame': row['switch_frame']}
    for plan in lists.values():
        plan['videos'] = list(plan['videos'].values())
        for video in plan['videos']:
            video['objects'] = list(video['objects'].values())
            for obj in video['objects']:
                obj['switches'] = list(obj['switches'].values())
    return lists


def read_results(dataset=None, *, seed=None, runs=None, visible_only=False, experiment_id=None):
    if not database_path().exists():
        return []
    selected = select_experiment(experiment_id) if experiment_id is not None else None
    identity = selected['experiment_id'] if selected is not None else experiment()[0]
    out = []
    with connect() as conn:
        for raw in conn.execute('SELECT * FROM results WHERE experiment_id=? ORDER BY result_id', (identity,)):
            if (dataset is not None and raw['dataset'] != dataset or seed is not None and raw['seed'] != seed
                    or runs is not None and raw['run_id'] > runs):
                continue
            rid = raw['result_id']
            row = json.loads(raw['metadata'])
            row.update(result_id=rid, experiment_id=identity)
            for phase, field in (('pre', 'pre_switch_frame_scores'), ('post', 'frame_scores')):
                query = 'SELECT * FROM frame_scores WHERE result_id=? AND phase=?'
                if visible_only:
                    query += ' AND has_gt=1 AND gt_visible=1'
                row[field] = []
                for raw_point in conn.execute(query + ' ORDER BY frame', (rid, phase)):
                    p = {k: raw_point[k] for k in raw_point.keys() if k not in ('result_id', 'phase')}
                    p.update(native_j=None, native_jf=None, recovery_j=None, recovery_jf=None,
                             native_reference_count=0, native_reference_ready=False)
                    row[field].append(p)
            row['frame_times'] = [{k: p[k] for k in ('frame', 'seconds', 'gpu_peak_mb')}
                                  for p in conn.execute('SELECT * FROM result_frame_times WHERE result_id=? ORDER BY frame', (rid,))]
            row['restoration_frame_scores'] = []
            for raw_point in conn.execute('SELECT * FROM restoration_frames WHERE result_id=? ORDER BY frame', (rid,)):
                p = {k: raw_point[k] for k in raw_point.keys() if k != 'result_id'}
                for flag in ('native_is_cond', 'target_is_cond'):
                    p[flag] = bool(p[flag]) if p[flag] is not None else None
                for raw_part in conn.execute('SELECT * FROM restoration_fields WHERE result_id=? AND frame=?', (rid, p['frame'])):
                    part = {k: raw_part[k] for k in raw_part.keys() if k not in ('result_id', 'frame', 'field')}
                    for shape in ('native_shape', 'target_shape'):
                        part[shape] = json.loads(part[shape]) if part[shape] is not None else None
                    p[raw_part['field']] = part
                row['restoration_frame_scores'].append(p)
            out.append(row)
    if not out:
        return []
    from evaluation.records import current_rows
    if selected is None:
        return current_rows(out)
    from evaluation.methods import METHODS
    unsupported = set(selected['baseline_revisions']) - {m.name for m in METHODS}
    if unsupported:
        raise ValueError(f'현재 집계 코드가 지원하지 않는 방법: {sorted(unsupported)}')
    return current_rows(out, configuration=selected['configuration'], revisions=selected['baseline_revisions'])


def visible_rows(rows):
    """추론 원점수를 변경하지 않고 기존 지표가 쓰는 가시 GT 표본을 만든다."""
    out = []
    for row in rows:
        result = dict(row)
        for field in ('pre_switch_frame_scores', 'frame_scores'):
            result[field] = [p for p in row[field] if p.get('has_gt', True) and p.get('gt_visible', True)]
        out.append(result)
    return out


def done_keys(dataset, *, videos=None, run_ids=None, seed=None):
    """재개 시 큰 프레임 행을 읽지 않는다. 결과는 자식 행과 함께 커밋된다."""
    from evaluation.records import iter_current_rows, row_key
    if not database_path().exists():
        return set()
    identity, _ = experiment()
    with connect() as conn:
        rows = (json.loads(r['metadata']) for r in conn.execute(
            'SELECT metadata FROM results WHERE experiment_id=? AND dataset=?', (identity, dataset)))
        return {row_key(r) for r in iter_current_rows(rows)
                if (videos is None or r['video'] in videos)
                and (run_ids is None or r['run_id'] in run_ids)
                and (seed is None or r['seed'] == seed)}


def save_analysis(rows, run_ids, statistic, seed, *, experiment_id=None):
    if not rows:
        raise ValueError('집계할 결과가 없습니다. 기존 분석/표는 변경하지 않습니다.')
    identities = {row['experiment_id'] for row in rows}
    if len(identities) != 1 or experiment_id is not None and identities != {experiment_id}:
        raise ValueError('집계 결과에 서로 다른 experiment_id가 섞였습니다.')
    identity = next(iter(identities))
    selected = select_experiment(identity)
    revision = selected['configuration']['native_memory_revision']
    if revision != restoration.REVISION:
        raise ValueError(f'지원하지 않는 복원율 정의 버전: {revision}')
    if any(row['baseline_revision'] != selected['baseline_revisions'][row['baseline']] for row in rows):
        raise ValueError('선택 실험의 최신 방법 revision과 집계 결과가 다릅니다.')
    definition = dict(experiment_id=identity, seed=seed, run_ids=list(run_ids), native_statistic=statistic,
                      recovery_statistic='ratio_of_means', restoration_revision=revision,
                      baseline_revisions=selected['baseline_revisions'])
    encoded = dumps(definition)
    analysis_id = hashlib.sha256(encoded.encode()).hexdigest()
    with connect() as conn:
        conn.execute('DELETE FROM analyses WHERE analysis_id=?', (analysis_id,))
        _insert(conn, 'analyses', dict(definition, analysis_id=analysis_id,
                                     run_ids=dumps(list(run_ids)), configuration=encoded))
        for row in rows:
            rid = row['result_id']
            for phase, field in (('pre', 'pre_switch_frame_scores'), ('post', 'frame_scores')):
                _insert_many(conn, 'recovery_frames', (dict(p, analysis_id=analysis_id, result_id=rid,
                             phase=phase, native_statistic=statistic) for p in row[field]))
            stats = {k: v for k, v in row.items() if k.startswith(('pre_', 'post_')) and k != 'pre_switch_frame_scores'}
            stats.update(restoration_summary=row.get('restoration_summary'),
                         r2_maskmem_features=row.get('r2_maskmem_features'), r2_obj_ptr=row.get('r2_obj_ptr'))
            _insert(conn, 'recovery_summaries', dict(stats, analysis_id=analysis_id, result_id=rid,
                    native_statistic=statistic, statistics=dumps(stats)))
    return analysis_id
