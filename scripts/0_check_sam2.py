"""[0] 설치된 SAM2 확인: Small/Base+ 기억 모양, 꺼냈다 넣어도 결과가 같은지.

    python scripts/0_check_sam2.py
    python scripts/0_check_sam2.py --dataset vost_val --video <영상 이름>
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from evaluation.data import load_dataset  # noqa: E402
from evaluation.data.common import object_ids  # noqa: E402
from model import sam2_check, sam2_runner  # noqa: E402


def ok(flag: bool) -> str:
    return "OK  " if flag else "문제"


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('1 이상의 정수가 필요합니다.')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lvos_v2_valid")
    parser.add_argument("--video", default=None, help="비우면 첫 영상")
    parser.add_argument("--cut", type=positive, default=settings.MEMORY_WINDOW + 2,
                        help="프롬프트 뒤 몇 프레임에서 끊을지 (기본값은 기억 정리 이후)")
    parser.add_argument("--after", type=positive, default=15, help="끊은 뒤 몇 프레임을 비교할지")
    args = parser.parse_args()

    # 재검사에 실패하면 이전 통과 기록을 재사용하지 않는다.
    sam2_check.report_path().unlink(missing_ok=True)
    videos = load_dataset(args.dataset)
    video = next((v for v in videos if args.video in (None, v.name)), None)
    if video is None:
        raise ValueError(f"검사할 영상이 없습니다: {args.dataset}/{args.video}")
    prompts = [(f, ids[0]) for f in sorted(video.mask_paths)
               if (ids := object_ids(video.read_labels(f)[0])) and f < video.num_frames - 1]
    if not prompts:
        raise ValueError("정답 객체와 이어 추적할 프레임이 있는 영상이 필요합니다.")
    start, obj_id = prompts[0]
    cut = min(start + args.cut, video.num_frames - 2)
    last = min(cut + args.after, video.num_frames - 1)
    print(f"영상 {args.dataset}/{video.name}, 객체 {obj_id}, 프롬프트 {start}, 끊는 곳 {cut}, 끝 {last}\n")

    runners = {key: sam2_runner.load_runner(key)
               for key in (settings.SOURCE_MODEL, settings.TARGET_MODEL)}
    infos, checks, roundtrips = {}, {}, {}
    for key, runner in runners.items():
        info = sam2_check.describe(runner, video, obj_id, start)
        infos[key] = info
        checks[f'{key}/state_keys'] = not info['missing_state_keys']
        checks[f'{key}/window'] = sam2_check.window_is_enough(info)
        entry = info['entry']
        checks[f'{key}/dtypes'] = (entry['maskmem_features'].dtype == torch.bfloat16
                                  and entry['obj_ptr'].dtype == torch.float32)
        print(f"[{key}] image_size={info['image_size']} num_maskmem={info['num_maskmem']} "
              f"max_obj_ptrs_in_encoder={info['max_obj_ptrs_in_encoder']} "
              f"hidden_dim={info['hidden_dim']} mem_dim={info['mem_dim']}")
        for name, shape in info["fields"].items():
            print(f"    {name:22s} {shape}")
        print(f"  {ok(not info['missing_state_keys'])} inference_state 필요한 키 "
              f"{'모두 있음' if not info['missing_state_keys'] else info['missing_state_keys']}")
        print(f"  {ok(sam2_check.window_is_enough(info))} MEMORY_WINDOW={settings.MEMORY_WINDOW} 이 "
              f"SAM2가 읽는 범위를 덮음\n")
        print(f"  {ok(checks[f'{key}/dtypes'])} spatial bf16 / pointer fp32\n")

    a, b = infos[settings.SOURCE_MODEL], infos[settings.TARGET_MODEL]
    mismatches = sam2_check.shape_mismatches(a, b)
    checks['shapes'] = not mismatches
    checks['pos_enc'] = sam2_check.same_pos_enc(a, b)
    print(f"{ok(not mismatches)} Small/Base+ 기억 칸 모양 같음" + (f": {mismatches}" if mismatches else ""))
    print(f"{ok(sam2_check.same_pos_enc(a, b))} 기억 위치 정보(maskmem_pos_enc) 값 같음\n")

    for key, runner in runners.items():
        r = sam2_check.roundtrip(runner, video, obj_id, start, cut, last)
        roundtrips[key] = r
        same = r['passed']
        checks[f'{key}/roundtrip'] = same
        print(f"{ok(same)} [{key}] 꺼냈다 넣기: {r['identical_frames']}/{r['frames']} 프레임 동일 "
              f"(요청 {r['expected_frames']}프레임, 가장 많이 다른 프레임 {r['max_diff_pixels']} 픽셀)")
    path = sam2_check.save_passed(checks, roundtrips,
                                 {'dataset': args.dataset, 'video': video.name, 'object': obj_id,
                                  'start': start, 'cut': cut, 'last': last})
    print(f"검사 통과 기록: {path}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
