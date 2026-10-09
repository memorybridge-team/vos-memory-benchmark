"""[2] 전체 방법을 기본 3회 평가. SQLite에 원자적으로 저장하고 중단 후 이어한다.

python scripts/2_evaluate.py --dataset vost_val
python scripts/2_evaluate.py --dataset vost_val --runs 1
python scripts/2_evaluate.py --dataset vost_val --run-id 2 --shard 0/2

Native는 각 회차·영상·객체에서 한 번 실행하여 25/50/75%와 모든 방법에 공유한다.
SQLite의 Native scalar/tensor 기준을 이어하기에서도 재사용한다. 낮은 성능/빈 예측은 제외하지 않는다.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings
import translator
from evaluation import native, records, store, runtime
from evaluation.data import load_dataset, load_video_list
from evaluation.evaluate_video import evaluate_object
from evaluation.methods import to_run
from model import sam2_runner


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('1 이상의 정수가 필요합니다.')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--max-videos', type=positive, default=None)
    parser.add_argument('--videos', nargs='*', default=None)
    parser.add_argument('--shard', default=None, help='i/n: 영상 분할')
    repeat = parser.add_mutually_exclusive_group()
    repeat.add_argument('--runs', type=positive, default=None, help='1~N회 평가 (기본 3)')
    repeat.add_argument('--run-id', type=positive, help='이 회차만 평가')
    parser.add_argument('--seed', type=int, default=settings.EVALUATION_SEED)
    parser.add_argument('--frame-prefetch', type=int, default=settings.FRAME_PREFETCH, help='선행 CPU 프레임 수, 0=비활성화')
    args = parser.parse_args()
    if args.frame_prefetch < 0:
        parser.error('--frame-prefetch는 0 이상이어야 합니다.')
    settings.FRAME_PREFETCH = args.frame_prefetch
    run_ids = [args.run_id] if args.run_id else range(1, (args.runs or settings.EVALUATION_RUNS) + 1)

    entries = load_video_list(args.dataset)['videos']
    if args.videos:
        entries = [e for e in entries if e['video'] in args.videos]
    if args.max_videos:
        entries = entries[:args.max_videos]
    if args.shard:
        try:
            i, n = map(int, args.shard.split('/'))
            if not 0 <= i < n:
                raise ValueError
        except ValueError:
            parser.error('--shard는 0 <= i < n인 i/n이어야 합니다.')
        entries = entries[i::n]
    methods = to_run()
    names = {e['video'] for e in entries}
    done = records.done_keys(args.dataset, videos=names, run_ids=run_ids, seed=args.seed)
    done_cases = {k[:4] for k in done}
    total = sum(len(e['objects']) for e in entries)
    tasks = []
    for run_id in run_ids:
        count = 0
        for entry in entries:
            for obj in entry['objects']:
                count += 1
                pending = {(sw['name'], sw['frame'], m.name)
                           for sw in obj['switches'] for m in methods
                           if (run_id, args.seed, entry['video'], obj['object'],
                               sw['name'], sw['frame'], m.name) not in done}
                if pending:
                    todo = [m for m in methods if any(name == m.name for _, _, name in pending)]
                    tasks.append((run_id, count, entry, obj, todo, pending))
    if not tasks:
        print('평가할 미완료 조건이 없습니다.')
        return

    # 빈 shard/완료한 평가에는 모델과 데이터 파일 목록을 로딩하지 않는다.
    active_names = {entry['video'] for _, _, entry, _, _, _ in tasks}
    videos = {v.name: v for v in load_dataset(args.dataset, names=active_names)}
    refs = native.load_references(args.dataset, videos=active_names, run_ids=run_ids, seed=args.seed)
    needs_native = False
    for run_id, _, entry, obj, _, _ in tasks:
        ref_key = native.case_key(videos[entry['video']], obj, run_id, args.seed)
        if ref_key not in refs:
            if (run_id, args.seed, entry['video'], obj['object']) in done_cases:
                raise ValueError(f'완료 결과의 Native 기준이 없습니다: {ref_key}. SQLite Native 기준을 복구하세요.')
            needs_native = True
    needs_gpu = needs_native or any(m.name != 'full_replay' for _,_,_,_,todo,_ in tasks for m in todo)
    with runtime.gpu_lease(settings.DEVICE, enabled=needs_gpu):
        if any(m.name == 'translator' for _, _, _, _, todo, _ in tasks for m in todo):
            translator.load()
        small = (sam2_runner.load_runner(settings.SOURCE_MODEL)
                 if any(m.name != 'full_replay' for _, _, _, _, todo, _ in tasks for m in todo) else None)
        base = (sam2_runner.load_runner(settings.TARGET_MODEL)
                if needs_native or any(m.name not in ('source_only', 'full_replay')
                                       for _, _, _, _, todo, _ in tasks for m in todo) else None)

        for run_id, count, entry, obj, todo, pending in tasks:
            out_path = store.database_path()
            video = videos[entry['video']]
            ref_key = native.case_key(video, obj, run_id, args.seed)

            def save_reference(ref, snapshots):
                store.save_native(ref, snapshots)
                refs[ref_key] = ref

            t0 = time.time()
            rows = evaluate_object(video, obj, small, base, todo, run_id=run_id, seed=args.seed,
                                   native_reference=refs.get(ref_key), save_native=save_reference,
                                   pending_conditions=pending)
            store.save_evaluations(rows)
            done.update(records.row_key(r) for r in rows)
            print(f'[회차 {run_id}, {count}/{total}] {video.name} 객체 {obj["object"]}: '
                  f'{len(rows)}줄, {time.time() - t0:.0f}초')
        print(f'결과: {out_path}')



if __name__ == '__main__':
    main()
