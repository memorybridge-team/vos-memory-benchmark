"""M3VOS (M3-VOS, CVPR 2025). 무시 영역 없음.

확인한 것 (huggingface.co/datasets/Lijiaxin0111/M3_VOS):
  - 정답 PNG 는 팔레트 모드, 값은 0(배경)과 객체 번호(1, 2, 3 ...)뿐. 영상 12개 × 3장을 열어 봤을 때 255 없음
  - 프레임 이름 7자리 (0000000.jpg), 평가 목록 ImageSets/val.txt
폴더 모양은 받은 곳에 따라 두 가지 — 둘 다 읽는다:
    Hugging Face    data/JPEGImages/<영상>/   data/Annotations/<영상>/   data/ImageSets/val.txt
    Google Drive    JPEGImages/<영상>/        Annotations/<영상>/        ImageSets/val.txt

확인: python -m benchmark.data.m3vos
"""

import json

from benchmark.data.common import dataset_root, print_first_video, read_names, videos_from_folders

IGNORE_VALUE = None


def load(split=None):
    root = dataset_root("m3vos")
    if (root / "data" / "JPEGImages").is_dir():
        root = root / "data"
    list_file = root / "ImageSets" / "val.txt"
    names = read_names(list_file) if list_file.exists() else None
    return videos_from_folders("m3vos", root / "JPEGImages", root / "Annotations",
                               names=names, ignore_value=IGNORE_VALUE, has_full_gt=True)


def load_labels(split=None) -> dict:
    """[추가] meta/all_phase_transition.json: 객체마다 변하기 전·후 상태와 변화 종류.
        {"0001_open_cup_1": {"1": {"before_state": "solid:non_particulate:rigid body",
                                   "after_state": "...", "phase transition": "separate"}}}
    객체 단위 → {영상: {객체 번호: ["상태 변화:separate", "상태:solid→liquid"]}}
    """
    root = dataset_root("m3vos")
    path = root / "meta" / "all_phase_transition.json"
    if not path.exists():
        return {}
    labels = {}
    for video, objects in json.loads(path.read_text(encoding="utf-8")).items():
        for obj, info in objects.items():
            before = info["before_state"].split(":")[0]
            after = info["after_state"].split(":")[0]
            labels.setdefault(video, {})[int(obj)] = [f"상태 변화:{info['phase transition']}",
                                                      f"상태:{before}→{after}"]
    return labels


if __name__ == "__main__":
    print_first_video(load())
