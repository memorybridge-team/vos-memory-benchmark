"""M3VOS (M3-VOS, CVPR 2025). 정답 PNG의 255 = 무시 영역 (객체가 아님, 채점에서 뺌).

확인한 것 (huggingface.co/datasets/Lijiaxin0111/M3_VOS):
  - 정답 PNG 는 팔레트 모드, 값은 0(배경)과 객체 번호(1, 2, 3 ...). 영상 12개 × 3장을 열어 봤을 때 255 없음
  - 팀 manifest·공식 reader 는 255 를 void 로 보고 객체에서 뺀다 → 우리도 무시 영역으로 둔다 (없으면 결과 그대로)
  - 프레임 이름 7자리 (0000000.jpg), 평가 목록 data/ImageSets/val.txt

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["m3vos"]>, 서버의 Hugging Face 받은 모양):
    data/JPEGImages/<영상>/   data/Annotations/<영상>/   data/ImageSets/val.txt   meta/

확인: python -m evaluation.data.m3vos
"""

import json

from evaluation.data.common import dataset_root, print_first_video, read_names, videos_from_folders

IGNORE_VALUE = 255


def load(split=None, *, names=None):
    data = dataset_root("m3vos") / "data"
    split_names = read_names(data / "ImageSets" / "val.txt")
    if names is not None:
        selected = set(names)
        split_names = [name for name in split_names if name in selected]
    return videos_from_folders("m3vos", data / "JPEGImages", data / "Annotations",
                               names=split_names,
                               ignore_value=IGNORE_VALUE)


def load_labels(split=None) -> dict:
    """[추가] meta/all_phase_transition.json: 객체마다 변하기 전·후 상태와 변화 종류.
        {"0001_open_cup_1": {"1": {"before_state": "solid:non_particulate:rigid body",
                                   "after_state": "...", "phase transition": "separate"}}}
    객체 단위 → {영상: {객체 번호: ["상태 변화:separate", "상태:solid→liquid"]}}
    """
    path = dataset_root("m3vos") / "meta" / "all_phase_transition.json"
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
