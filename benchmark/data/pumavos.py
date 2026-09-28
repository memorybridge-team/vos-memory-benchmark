"""PUMaVOS. 모든 프레임에 정답, 무시 영역 없음.

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["pumavos"]>) — 서버와 다르면 여기만 고친다:
    JPEGImages/<영상>/*.jpg
    Annotations/<영상>/*.png

확인: python -m benchmark.data.pumavos
"""

from benchmark.data.common import dataset_root, print_first_video, videos_from_folders

IGNORE_VALUE = None


def load(split=None):
    root = dataset_root("pumavos")
    return videos_from_folders("pumavos", root / "JPEGImages", root / "Annotations",
                               ignore_value=IGNORE_VALUE, has_full_gt=True)


if __name__ == "__main__":
    print_first_video(load())
