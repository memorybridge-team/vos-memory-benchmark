"""[0] 설치된 SAM2 확인: Small/Base+ 기억 모양, 꺼냈다 넣어도 결과가 같은지.

    python scripts/0_check_sam2.py
    python scripts/0_check_sam2.py --dataset lvos_v2_train --video <영상 이름>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from evaluation.data import load_dataset  # noqa: E402
from evaluation.data.common import object_ids  # noqa: E402
from model import sam2_check, sam2_runner  # noqa: E402


def ok(flag: bool) -> str:
    return "OK  " if flag else "문제"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="lvos_v2_train")
    parser.add_argument("--video", default=None, help="비우면 첫 영상")
    parser.add_argument("--cut", type=int, default=10, help="프롬프트 뒤 몇 프레임에서 끊을지")
    parser.add_argument("--after", type=int, default=15, help="끊은 뒤 몇 프레임을 비교할지")
    args = parser.parse_args()

    videos = load_dataset(args.dataset)
    video = next(v for v in videos if args.video in (None, v.name))
    start = min(video.mask_paths)
    obj_id = object_ids(video.read_labels(start)[0])[0] #[0]: 물체 번호 지도/[1] : 무시영역 - 검증용 코드
    cut = min(start + args.cut, video.num_frames - 2)
    last = min(cut + args.after, video.num_frames - 1)
    print(f"영상 {args.dataset}/{video.name}, 객체 {obj_id}, 프롬프트 {start}, 끊는 곳 {cut}, 끝 {last}\n")

    runners = {key: sam2_runner.load_runner(key)
               for key in (settings.SOURCE_MODEL, settings.TARGET_MODEL)}
    infos = {}
    for key, runner in runners.items():
        info = sam2_check.describe(runner, video, obj_id, start)
        infos[key] = info
        print(f"[{key}] image_size={info['image_size']} num_maskmem={info['num_maskmem']} "
              f"max_obj_ptrs_in_encoder={info['max_obj_ptrs_in_encoder']} "
              f"hidden_dim={info['hidden_dim']} mem_dim={info['mem_dim']}")
        for name, shape in info["fields"].items():
            print(f"    {name:22s} {shape}")
        print(f"  {ok(not info['missing_state_keys'])} inference_state 필요한 키 "
              f"{'모두 있음' if not info['missing_state_keys'] else info['missing_state_keys']}")
        print(f"  {ok(sam2_check.window_is_enough(info))} MEMORY_WINDOW={settings.MEMORY_WINDOW} 이 "
              f"SAM2가 읽는 범위를 덮음\n")

    a, b = infos[settings.SOURCE_MODEL], infos[settings.TARGET_MODEL]
    mismatches = sam2_check.shape_mismatches(a, b)
    print(f"{ok(not mismatches)} Small/Base+ 기억 칸 모양 같음" + (f": {mismatches}" if mismatches else ""))
    print(f"{ok(sam2_check.same_pos_enc(a, b))} 기억 위치 정보(maskmem_pos_enc) 값 같음\n")

    for key, runner in runners.items():
        r = sam2_check.roundtrip(runner, video, obj_id, start, cut, last)
        same = r["identical_frames"] == r["frames"]
        print(f"{ok(same)} [{key}] 꺼냈다 넣기: {r['identical_frames']}/{r['frames']} 프레임 동일 "
              f"(가장 많이 다른 프레임 {r['max_diff_pixels']} 픽셀)")


if __name__ == "__main__":
    main()
