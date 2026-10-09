"""회차별 Native 원점수·비용과 전환 시점 기억 tensor를 SQLite에 보존한다."""

import io
import re
from pathlib import Path
from uuid import uuid4

import torch

import settings
from evaluation import cost, store
from evaluation.scoring.jf import FrameScore
from evaluation.scoring import restoration


def reference_path(dataset, run_id, shard=None):
    suffix = f'.shard{shard.replace("/", "of")}' if shard else ''
    return Path(settings.OUTPUT_ROOT) / 'native' / f'{dataset}.run{run_id}{suffix}.jsonl'


def case_key(video, obj, run_id, seed):
    return (run_id, seed, video.name, obj['object'], obj['start'], obj['end'],
            tuple((s['name'], s['frame']) for s in obj['switches']))


def reference_key(ref):
    return (ref['run_id'], ref['seed'], ref['video'], ref['object'], ref['start'], ref['end'],
            tuple((s['name'], s['frame']) for s in ref['switches']))


def load_references(dataset, *, videos=None, run_ids=None, seed=None):
    store.import_legacy(dataset)
    refs = {}
    for ref in store.iter_references(dataset, videos=videos, run_ids=run_ids, seed=seed):
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
        'scores': [{'frame': f, 'j': s.j, 'f': s.f, 'gt_visible': s.gt_visible}
                   for f, s in sorted(run.scores.items())],
        'costs': {str(sw['frame']): cost.cost_columns(run, sw['frame']) for sw in obj['switches']},
    }


def scores(ref):
    return {p['frame']: FrameScore(p['j'], p['f'], p['gt_visible']) for p in ref['scores']}


def memory_path(ref):
    identity = ref['native_reference_id']
    if not isinstance(identity, str) or re.fullmatch(r'[0-9a-f]{32}', identity) is None:
        raise ValueError('Native 기준 ID가 유효하지 않습니다.')
    return Path(settings.OUTPUT_ROOT) / 'native_memory' / f'{identity}.pt'


def save_memories(ref, snapshots):
    """Native scalar와 tensor BLOB을 함께 SQLite에 저장한다."""
    store.save_native(ref, snapshots)


def load_memories(ref):
    path = memory_path(ref)  # ID 검증 및 이전 .pt 호환
    blob = store.memory_blob(ref['native_reference_id'])
    if blob is None and not path.exists():
        raise ValueError(f"Native 기억 기준이 DB/이전 .pt에 없습니다: {ref['native_reference_id']}")
    source = io.BytesIO(blob) if blob is not None else path
    payload = torch.load(source, map_location='cpu', weights_only=True)
    if (payload.get('native_reference_id') != ref['native_reference_id']
            or payload.get('native_memory_revision') != restoration.REVISION):
        raise ValueError(f'Native 기억 기준의 ID/버전이 다릅니다: {path}')
    snapshots = payload['snapshots']
    if set(snapshots) != {sw['frame'] for sw in ref['switches']}:
        raise ValueError(f'Native 기억 기준의 전환 프레임이 다릅니다: {path}')
    return snapshots
