"""VOST val. 정답 PNG의 255 = 무시 영역 (채점에서 뺌).

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["vost"]>) — 서버와 다르면 여기만 고친다:
    JPEGImages/<영상>/*.jpg
    Annotations/<영상>/*.png
    ImageSets/val.txt           val 영상 이름 목록

확인: python -m benchmark.data.vost
"""

from benchmark.data.common import dataset_root, print_first_video, read_names, videos_from_folders

IGNORE_VALUE = 255


def load(split: str = "val"):
    root = dataset_root("vost")
    names = read_names(root / "ImageSets" / f"{split}.txt")
    return videos_from_folders(f"vost_{split}", root / "JPEGImages", root / "Annotations",
                               names=names, ignore_value=IGNORE_VALUE, has_full_gt=True)


if __name__ == "__main__":
    print_first_video(load("val"))
