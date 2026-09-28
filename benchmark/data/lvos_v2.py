"""LVOS v2. train / valid 모두 모든 프레임에 정답.

객체가 영상 중간에 처음 나타날 수 있다 → 객체마다 처음 보인 프레임에서 시작 (1_make_video_list.py).

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["lvos_v2"]>) — 서버와 다르면 여기만 고친다:
    train/JPEGImages/<영상>/*.jpg    train/Annotations/<영상>/*.png
    valid/JPEGImages/<영상>/*.jpg    valid/Annotations/<영상>/*.png

확인: python -m benchmark.data.lvos_v2
"""

from benchmark.data.common import dataset_root, print_first_video, videos_from_folders

IGNORE_VALUE = None


def load(split: str):
    root = dataset_root("lvos_v2") / split
    return videos_from_folders(f"lvos_v2_{split}", root / "JPEGImages", root / "Annotations",
                               ignore_value=IGNORE_VALUE, has_full_gt=True)


if __name__ == "__main__":
    print_first_video(load("train"))
    print_first_video(load("valid"))
