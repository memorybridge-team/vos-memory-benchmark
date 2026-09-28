"""[3] 데이터셋 하나 평가: 목록의 영상 × 객체 × 전환 시점 × 비교군.

    python scripts/3_evaluate.py --dataset lvos_v2_train --part dev --max-videos 5   # 개발 중
    python scripts/3_evaluate.py --dataset lvos_v2_valid                             # 최종
    python scripts/3_evaluate.py --dataset mosev2_valid     # 채점 대신 제출 zip 28개를 만듦

결과: outputs/records/<데이터셋>.jsonl (끊겨도 다시 실행하면 이어서 진행)
MOSEv2 valid: outputs/mosev2/submissions/*.zip → 사람이 서버에 제출 → 4_mosev2_scores.py
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from benchmark import moment_stats, records  # noqa: E402
from benchmark.data import load_dataset, load_video_list  # noqa: E402
from benchmark.evaluate_video import evaluate_object  # noqa: E402
from benchmark.model import sam2_runner  # noqa: E402
from benchmark.scoring import mosev2_server  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--part", choices=["fit", "dev", "all"], default="all",
                        help="train 데이터셋에서 fit / dev 중 어느 쪽을 쓸지")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--videos", nargs="*", default=None, help="이 영상들만")
    args = parser.parse_args()

    video_list = load_video_list(args.dataset)
    entries = [e for e in video_list["videos"]
               if args.part == "all" or e["part"] == args.part]
    if args.videos:
        entries = [e for e in entries if e["video"] in args.videos]
    if args.max_videos:
        entries = entries[:args.max_videos]
    videos = {v.name: v for v in load_dataset(args.dataset)}

    stats = moment_stats.load()
    small = sam2_runner.load_runner(settings.SOURCE_MODEL)
    base = sam2_runner.load_runner(settings.TARGET_MODEL)
    saver = None if video_list["has_full_gt"] else mosev2_server.MaskSaver()

    out_path = records.records_path(args.dataset)
    done = records.done_objects(out_path)
    total = sum(len(e["objects"]) for e in entries)
    count = 0
    for entry in entries:
        video = videos[entry["video"]]
        for obj in entry["objects"]:
            count += 1
            if (video.name, obj["object"]) in done:
                continue
            t0 = time.time()
            rows = evaluate_object(video, obj, small, base, stats, part=entry["part"], saver=saver)
            records.append_rows(out_path, rows)
            print(f"[{count}/{total}] {video.name} 객체 {obj['object']}: 줄 {len(rows)}개, "
                  f"{time.time() - t0:.0f}초")
    print(f"결과: {out_path}")

    if saver is not None:
        # 제출 파일은 전체 목록 기준으로 만든다 (부분 실행이면 안 돈 객체는 빈 마스크로 들어감).
        mosev2_server.build_submissions(video_list, videos, saver)


if __name__ == "__main__":
    main()
