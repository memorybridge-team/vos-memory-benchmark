"""MOSEv2.

train = 모든 프레임에 정답 / valid = 첫 프레임에만 정답 (점수는 서버에서 받음).

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["mosev2"]>) — 서버와 다르면 여기만 고친다:
    train/JPEGImages/<영상>/*.jpg    train/Annotations/<영상>/*.png
    valid/JPEGImages/<영상>/*.jpg    valid/Annotations/<영상>/*.png

확인: python -m benchmark.data.mosev2
"""

from benchmark.data.common import dataset_root, print_first_video, videos_from_folders

IGNORE_VALUE = None


def load(split: str):
    root = dataset_root("mosev2") / split
    return videos_from_folders(f"mosev2_{split}", root / "JPEGImages", root / "Annotations",
                               ignore_value=IGNORE_VALUE,
                               has_full_gt=(split == "train"))


if __name__ == "__main__":
    print_first_video(load("train"))
    print_first_video(load("valid"))
