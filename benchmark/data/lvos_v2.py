"""LVOS v2. train / val 모두 정답 공개 (test 는 첫 프레임뿐이라 쓰지 않음).

공식 README 기준 (github.com/LingyiHongfd/LVOS):
  - 나누기 폴더 이름은 train / val / test ("valid" 아님)
  - 프레임 이름 8자리 (00000001.jpg ...), 6fps 로 뽑은 프레임과 정답
객체가 영상 중간에 처음 나타날 수 있다 → 객체마다 처음 보인 프레임에서 시작 (1_make_video_list.py).

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["lvos_v2"]>) — 서버와 다르면 여기만 고친다:
    train/JPEGImages/<영상>/*.jpg    train/Annotations/<영상>/*.png
    val/JPEGImages/<영상>/*.jpg      val/Annotations/<영상>/*.png

확인: python -m benchmark.data.lvos_v2
"""

from benchmark.data.common import dataset_root, print_first_video, videos_from_folders

IGNORE_VALUE = None
SPLIT_FOLDERS = {"train": "train", "valid": "val"}   # 우리 이름 → 실제 폴더 이름


def load(split: str):
    root = dataset_root("lvos_v2") / SPLIT_FOLDERS[split]
    return videos_from_folders(f"lvos_v2_{split}", root / "JPEGImages", root / "Annotations",
                               ignore_value=IGNORE_VALUE, has_full_gt=True)


if __name__ == "__main__":
    print_first_video(load("train"))
    print_first_video(load("valid"))
