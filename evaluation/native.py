"""회차별 Native 원점수·비용은 SQLite, 전환 시점 기억 기준은 CPU tensor 파일에 보존한다."""

import os
import re
from pathlib import Path
from uuid import uuid4

import torch

import settings
from evaluation import cost
from evaluation.scoring.jf import FrameScore
from evaluation.scoring import restoration


def case_key(video, obj, run_id, seed):
    return (run_id, seed, video.name, obj['object'], obj['start'], obj['end'],
            tuple((s['name'], s['frame']) for s in obj['switches']))


def reference_key(ref):
    return (ref['run_id'], ref['seed'], ref['video'], ref['object'], ref['start'], ref['end'],
            tuple((s['name'], s['frame']) for s in ref['switches']))


def load_references(dataset, *, videos=None, run_ids=None, seed=None):
    refs = {}
    selected = None if videos is None else set(videos)
    runs = None if run_ids is None else set(run_ids)
    from evaluation.store import load_native
    for ref in load_native(dataset, videos=selected, run_ids=runs, seed=seed):
        key = reference_key(ref)
        if key in refs and refs[key]['native_reference_id'] != ref['native_reference_id']:
            raise ValueError(f'동일 조건의 Native 기준이 둘 이상입니다: {key}')
        refs[key] = ref
    return refs


def pack(video, obj, run, run_id, seed):
    return {
        'evaluation_revision': settings.EVALUATION_REVISION,
        'runtime_revision': settings.EVALUATION_RUNTIME_REVISION,
        'native_reference_id': uuid4().hex,
        'native_memory_revision': restoration.REVISION,
        'dataset': video.dataset, 'video': video.name, 'object': obj['object'],
        'start': obj['start'], 'end': obj['end'], 'switches': obj['switches'],
        'run_id': run_id, 'seed': seed,
        'cpu_profile': run.cpu_profile,
        'scores': [score_point(f, run.scores.get(f)) for f in range(obj['start'], obj['end'] + 1)],
        'frame_times': [{'frame': f, 'seconds': run.times.get(f), 'gpu_peak_mb': run.gpu_peaks.get(f)}
                        for f in range(obj['start'], obj['end'] + 1)],
        'costs': {str(sw['frame']): cost.cost_columns(run, sw['frame']) for sw in obj['switches']},
    }


def score_point(frame, score):
    return {'frame': frame, 'has_gt': score is not None,
            'gt_visible': score.gt_visible if score is not None else None,
            'j': score.j if score is not None else None,
            'f': score.f if score is not None else None,
            'jf': score.jf if score is not None else None}


def scores(ref):
    return {p['frame']: FrameScore(p['j'], p['f'], p['gt_visible']) for p in ref['scores'] if p['j'] is not None}


def memory_path(ref):
    identity = ref['native_reference_id']
    if not isinstance(identity, str) or re.fullmatch(r'[0-9a-f]{32}', identity) is None:
        raise ValueError('Native 기준 ID가 유효하지 않습니다.')
    return Path(settings.OUTPUT_ROOT) / 'native_memory' / f'{identity}.pt'


def save_memories(ref, snapshots):
    """전환 시점 CPU 기억 snapshot을 원자적으로 저장한 뒤 Native DB 기록을 커밋하게 한다."""
    path = memory_path(ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pt.tmp')
    payload = {'native_reference_id': ref['native_reference_id'],
               'native_memory_revision': restoration.REVISION, 'snapshots': snapshots}
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_memories(ref):
    path = memory_path(ref)
    if not path.exists():
        raise ValueError(f'Native 기억 기준 파일이 없습니다: {path}. outputs/native_memory를 복구하세요.')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if (payload.get('native_reference_id') != ref['native_reference_id']
            or payload.get('native_memory_revision') != restoration.REVISION):
        raise ValueError(f'Native 기억 기준의 ID/버전이 다릅니다: {path}')
    snapshots = payload['snapshots']
    if set(snapshots) != {sw['frame'] for sw in ref['switches']}:
        raise ValueError(f'Native 기억 기준의 전환 프레임이 다릅니다: {path}')
    return snapshots
