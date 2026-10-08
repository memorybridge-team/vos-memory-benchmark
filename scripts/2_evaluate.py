"""[2] 데이터셋 하나 평가: 목록의 영상 × 객체 × 전환 시점 × 방법 전부 (비교군 · 본 모델 한 번에).

    python scripts/2_evaluate.py --dataset lvos_v2_valid --max-videos 5   # 개발 중
    python scripts/2_evaluate.py --dataset lvos_v2_valid                  # 검증
    python scripts/2_evaluate.py --dataset vost_val                       # 평가 (m3vos, pumavos 도)
    CUDA_VISIBLE_DEVICES=1 python scripts/2_evaluate.py --dataset vost_val --shard 1/2   # GPU 2개 중 두 번째

결과: outputs/records/<데이터셋>.jsonl (끊겨도 다시 실행하면 이어서 진행)
      --shard i/n 이면 끝에 .shard{i}of{n} 이 붙는다 — 3_make_tables.py 가 records/*.jsonl 을 모두 읽어 합친다
      끝난 (영상, 객체, 전환, 방법) 은 이 데이터셋의 결과 파일 전부에서 찾는다
      → GPU 수를 바꾸거나, 예전에 본 모델만 따로 돌린 결과(<데이터셋>.model.jsonl)가 있어도 이어서 진행
Full Replay와 Source-only는 객체마다 한 번씩 실행하고 각 전환 뒤 구간을 잘라 쓴다.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
import translator  # noqa: E402
from evaluation import records  # noqa: E402
from evaluation.data import load_dataset, load_video_list  # noqa: E402
from evaluation.evaluate_video import evaluate_object  # noqa: E402
from evaluation.methods import to_run  # noqa: E402
from model import sam2_runner  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--videos", nargs="*", default=None, help="이 영상들만")
    parser.add_argument("--shard", default=None, help="i/n: 영상을 n 묶음으로 나눠 i번째만 (GPU 여러 개)")
    args = parser.parse_args()

    entries = load_video_list(args.dataset)["videos"]
    if args.videos:
        entries = [e for e in entries if e["video"] in args.videos]
    if args.max_videos:
        entries = entries[:args.max_videos]
    name = args.dataset
    if args.shard:      # 영상을 번갈아 n 묶음으로 나눠 i번째만 돌리고, 결과도 따로 저장
        i, n = map(int, args.shard.split("/"))
        entries = entries[i::n]
        name += f".shard{i}of{n}"
    out_path = records.records_path(args.dataset).with_name(f"{name}.jsonl")
    videos = {v.name: v for v in load_dataset(args.dataset)}

    translator.load()
    small = sam2_runner.load_runner(settings.SOURCE_MODEL)
    base = sam2_runner.load_runner(settings.TARGET_MODEL)

    methods = to_run()
    done = records.done_keys(args.dataset)
    total = sum(len(e["objects"]) for e in entries)
    count = 0
    for entry in entries:
        video = videos[entry["video"]]
        for obj in entry["objects"]:
            count += 1
            todo = [m for m in methods if any(
                (video.name, obj["object"], sw["name"], sw["frame"], m.name) not in done
                for sw in obj["switches"])]
            if not todo:
                continue
            t0 = time.time()
            rows = evaluate_object(video, obj, small, base, todo)
            rows = [r for r in rows if records.row_key(r) not in done]
            records.append_rows(out_path, rows)
            done.update(records.row_key(r) for r in rows)
            print(f"[{count}/{total}] {video.name} 객체 {obj['object']}: 줄 {len(rows)}개, "
                  f"{time.time() - t0:.0f}초")
    print(f"결과: {out_path}")


if __name__ == "__main__":
    main()
