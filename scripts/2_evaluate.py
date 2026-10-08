"""[2] 전체 방법을 기본 3회 평가. JSONL로 회차별 저장하고 중단 후 이어한다.

python scripts/2_evaluate.py --dataset vost_val
python scripts/2_evaluate.py --dataset vost_val --runs 1
python scripts/2_evaluate.py --dataset vost_val --run-id 2 --shard 0/2

Native는 각 회차·영상·객체에서 한 번 실행하여 50/75%와 모든 방법에 공유한다.
outputs/native/의 기준 파일을 이어하기에서도 재사용한다. 낮은 성능/빈 예측은 제외하지 않는다.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings
import translator
from evaluation import native, records
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
    args = parser.parse_args()
    run_ids = [args.run_id] if args.run_id else range(1, (args.runs or settings.EVALUATION_RUNS) + 1)

    entries = load_video_list(args.dataset)['videos']
    if args.videos:
        entries = [e for e in entries if e['video'] in args.videos]
    if args.max_videos:
        entries = entries[:args.max_videos]
    suffix = ''
    if args.shard:
        try:
            i, n = map(int, args.shard.split('/'))
            if not 0 <= i < n:
                raise ValueError
        except ValueError:
            parser.error('--shard는 0 <= i < n인 i/n이어야 합니다.')
        entries = entries[i::n]
        suffix = f'.shard{i}of{n}'
    videos = {v.name: v for v in load_dataset(args.dataset)}
    translator.load()
    small = sam2_runner.load_runner(settings.SOURCE_MODEL)
    base = sam2_runner.load_runner(settings.TARGET_MODEL)
    methods = to_run()
    done = records.done_keys(args.dataset)
    refs = native.load_references(args.dataset)
    total = sum(len(e['objects']) for e in entries)

    for run_id in run_ids:
        out_path = records.records_path(args.dataset).with_name(f'{args.dataset}.run{run_id}{suffix}.jsonl')
        ref_path = native.reference_path(args.dataset, run_id, args.shard)
        count = 0
        for entry in entries:
            video = videos[entry['video']]
            for obj in entry['objects']:
                count += 1
                todo = [m for m in methods if any(
                    (run_id, args.seed, video.name, obj['object'], sw['name'], sw['frame'], m.name) not in done
                    for sw in obj['switches'])]
                if not todo:
                    continue
                ref_key = native.case_key(video, obj, run_id, args.seed)
                existing = refs.get(ref_key)
                # 결과만 있고 기준 파일이 유실되었다면 새 분모를 조용히 섞지 않는다.
                has_done = any(k[:4] == (run_id, args.seed, video.name, obj['object']) for k in done)
                if has_done and existing is None:
                    raise ValueError(f'완료 결과의 Native 기준이 없습니다: {ref_key}. outputs/native를 복구하세요.')

                def save_reference(ref):
                    records.append_rows(ref_path, [ref])
                    refs[ref_key] = ref

                t0 = time.time()
                rows = evaluate_object(video, obj, small, base, todo, run_id=run_id, seed=args.seed,
                                       native_reference=existing, save_native=save_reference)
                rows = [r for r in rows if records.row_key(r) not in done]
                records.append_rows(out_path, rows)
                done.update(records.row_key(r) for r in rows)
                print(f'[회차 {run_id}, {count}/{total}] {video.name} 객체 {obj["object"]}: '
                      f'{len(rows)}줄, {time.time() - t0:.0f}초')
        print(f'결과: {out_path}')


if __name__ == '__main__':
    main()
