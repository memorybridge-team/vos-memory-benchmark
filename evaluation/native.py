"""회차별 Native 원점수·비용은 JSONL, 전환 시점 기억 기준은 CPU tensor 파일에 보존한다."""

import os
import re
from pathlib import Path
from uuid import uuid4

import torch

import settings
from evaluation import cost, records
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


def load_references(dataset):
    refs = {}
    for path in sorted((Path(settings.OUTPUT_ROOT) / 'native').glob(f'{dataset}.run*.jsonl')):
        for ref in records.read_rows(path):
            if ref.get('evaluation_revision') == settings.EVALUATION_REVISION and ref['dataset'] == dataset:
                key = reference_key(ref)
                if key in refs and refs[key]['native_reference_id'] != ref['native_reference_id']:
                    raise ValueError(f'동일 조건의 Native 기준이 둘 이상입니다: {key}')
                refs[key] = ref
    return refs


def pack(video, obj, run, run_id, seed):
    return {
        'evaluation_revision': settings.EVALUATION_REVISION,
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
    """전환 시점 CPU 기억 snapshot을 원자적으로 저장한 뒤 Native JSONL을 쓰게 한다."""
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
