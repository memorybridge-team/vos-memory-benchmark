"""VOST val. 정답 PNG의 255 = 무시 영역 (채점에서 뺌).

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["vost"]>) — 서버와 다르면 여기만 고친다:
    JPEGImages/<영상>/*.jpg
    Annotations/<영상>/*.png
    ImageSets/val.txt           val 영상 이름 목록

확인: python -m evaluation.data.vost
"""

from evaluation.data.common import dataset_root, print_first_video, read_names, videos_from_folders

IGNORE_VALUE = 255


def load(split: str = "val"):
    root = dataset_root("vost")
    names = read_names(root / "ImageSets" / f"{split}.txt")
    return videos_from_folders(f"vost_{split}", root / "JPEGImages", root / "Annotations",
                               names=names, ignore_value=IGNORE_VALUE)


def load_labels(split: str = "val") -> dict:
    """[추가] 공식 라벨 파일은 없다 (VOST.zip 안에 목록·영상·정답뿐).
    대신 공식 영상 이름이 '<번호>_<동작>_<물체>' (예: 3545_break_egg) 라서 동작을 라벨로 쓴다.
    영상 단위 → {영상: {"*": ["변형:break"]}}
    """
    labels = {}
    for name in read_names(dataset_root("vost") / "ImageSets" / f"{split}.txt"):
        parts = name.split("_")
        if len(parts) >= 3:
            labels[name] = {"*": [f"변형:{parts[1]}"]}
    return labels


if __name__ == "__main__":
    print_first_video(load("val"))
