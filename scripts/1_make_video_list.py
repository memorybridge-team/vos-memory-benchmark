"""[1] 데이터셋마다 영상·객체·전환 시점 목록을 만들어 파일로 고정한다.

    python scripts/1_make_video_list.py                          # 전부
    python scripts/1_make_video_list.py --datasets lvos_v2_valid vost_val

결과: outputs/lists/<데이터셋>.json
  객체 시작 = 정답에서 처음 보인 프레임, 끝 = 영상 마지막 프레임.
  [추가] 공식 라벨이 있는 데이터셋은 객체마다 "extra_labels" (evaluation/scoring/extra_groups.py).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from evaluation.data import DATASETS, labels_of, load_dataset, load_labels, video_list_path  # noqa: E402
from evaluation.data.common import object_ids  # noqa: E402
from evaluation.switches import switch_points  # noqa: E402


def scan_visibility(video) -> dict[int, dict[int, bool]]:
    """정답 PNG를 모두 읽어 {객체: {정답 프레임: 보이는가}} 를 만든다."""
    present = {}
    for frame in sorted(video.mask_paths):
        labels, _ = video.read_labels(frame)
        present[frame] = set(object_ids(labels))
    all_ids = sorted(set().union(*present.values())) if present else []
    return {obj: {f: obj in ids for f, ids in present.items()} for obj in all_ids}


def make_list(dataset: str) -> dict:
    videos = load_dataset(dataset)
    labels = load_labels(dataset)          # [추가] 공식 라벨 (없는 데이터셋은 빈 dict)
    entries, skipped = [], 0
    for i, video in enumerate(videos, 1):
        end = video.num_frames - 1
        objects = []
        for obj_id, visible in scan_visibility(video).items():
            start = min(f for f, v in visible.items() if v)
            if end - start < settings.MIN_TRACK_FRAMES:
                skipped += 1
                continue
            objects.append({"object": obj_id, "start": start, "end": end,
                            "switches": switch_points(start, end),
                            "extra_labels": labels_of(labels, video.name, obj_id)})
        entries.append({"video": video.name, "num_frames": video.num_frames, "objects": objects})
        if i % 50 == 0 or i == len(videos):
            print(f"  {dataset}: {i}/{len(videos)}")
    return {"dataset": dataset, "evaluation_revision": settings.VIDEO_LIST_REVISION,
            "switch_basis": "object", "switch_fractions": list(settings.SWITCH_FRACTIONS),
            "skipped_short_objects": skipped, "videos": entries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    for dataset in args.datasets:
        data = make_list(dataset)
        path = video_list_path(dataset)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        n_obj = sum(len(v["objects"]) for v in data["videos"])
        print(f"{dataset}: 영상 {len(data['videos'])}, 객체 {n_obj}, "
              f"짧아서 뺀 객체 {data['skipped_short_objects']} → {path}")


if __name__ == "__main__":
    main()
