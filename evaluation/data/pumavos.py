"""PUMaVOS. 모든 프레임에 정답, 무시 영역 없음.

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["pumavos"]>) — 서버와 다르면 여기만 고친다:
    JPEGImages/<영상>/*.jpg
    Annotations/<영상>/*.png
※ 서버에서는 extracted/PUBLIC_PUMaVOS/ 아래에 이 두 폴더가 있다 (2026-10-05 확인).

확인: python -m evaluation.data.pumavos
"""

from evaluation.data.common import dataset_root, print_first_video, videos_from_folders

IGNORE_VALUE = None


def load(split=None):
    root = dataset_root("pumavos")
    return videos_from_folders("pumavos", root / "JPEGImages", root / "Annotations",
                               ignore_value=IGNORE_VALUE)


if __name__ == "__main__":
    print_first_video(load())
