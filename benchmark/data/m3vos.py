"""M3VOS. 정답 PNG의 255 = 무시 영역 (채점에서 뺌).

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["m3vos"]>) — 서버와 다르면 여기만 고친다:
    JPEGImages/<영상>/*.jpg
    Annotations/<영상>/*.png

확인: python -m benchmark.data.m3vos
"""

from benchmark.data.common import dataset_root, print_first_video, videos_from_folders

IGNORE_VALUE = 255


def load(split=None):
    root = dataset_root("m3vos")
    return videos_from_folders("m3vos", root / "JPEGImages", root / "Annotations",
                               ignore_value=IGNORE_VALUE, has_full_gt=True)


if __name__ == "__main__":
    print_first_video(load())
