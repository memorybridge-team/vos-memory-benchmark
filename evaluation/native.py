"""회차별 Native 원점수와 전환 비용을 JSONL에 보존한다. 이어할 때 분모를 재실행하지 않는다."""

from pathlib import Path
from uuid import uuid4

import settings
from evaluation import cost, records
from evaluation.scoring.jf import FrameScore


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
        'dataset': video.dataset, 'video': video.name, 'object': obj['object'],
        'start': obj['start'], 'end': obj['end'], 'switches': obj['switches'],
        'run_id': run_id, 'seed': seed,
        'scores': [{'frame': f, 'j': s.j, 'f': s.f, 'gt_visible': s.gt_visible}
                   for f, s in sorted(run.scores.items())],
        'costs': {str(sw['frame']): cost.cost_columns(run, sw['frame']) for sw in obj['switches']},
    }


def scores(ref):
    return {p['frame']: FrameScore(p['j'], p['f'], p['gt_visible']) for p in ref['scores']}
