"""LVOS v2 valid (검증). 모든 프레임에 정답 공개.

공식 README 기준 (github.com/LingyiHongfd/LVOS):
  - 나누기 폴더 이름은 train / val / test ("valid" 아님)
  - 프레임 이름 8자리 (00000001.jpg ...), 6fps 로 뽑은 프레임과 정답
객체가 영상 중간에 처음 나타날 수 있다 → 객체마다 처음 보인 프레임에서 시작 (1_make_video_list.py).

폴더 모양 (DATA_ROOT/<DATA_FOLDERS["lvos_v2"]>) — 서버와 다르면 여기만 고친다:
    val/JPEGImages/<영상>/*.jpg      val/Annotations/<영상>/*.png   (val 대신 valid 여도 됨)

확인: python -m evaluation.data.lvos_v2
"""

import json

from evaluation.data.common import dataset_root, print_first_video, videos_from_folders

IGNORE_VALUE = None
SPLIT_FOLDERS = {"valid": ("val", "valid")}   # 우리 이름 → 실제 폴더 이름 후보
# README 는 val, 배포 zip 은 valid.zip, 공식 평가 코드는 valid_meta.json → 있는 쪽을 쓴다


def _split_root(split: str):
    base = dataset_root("lvos_v2")
    for name in SPLIT_FOLDERS[split]:
        if (base / name).is_dir():
            return base / name
    return base / SPLIT_FOLDERS[split][0]

# 공식 영상 속성 13종 (LVOS 논문 Table II)
ATTRIBUTES = {
    "BC": "배경 혼동", "DEF": "모양 변형", "MB": "움직임 흐림", "FM": "빠른 움직임",
    "LR": "작은 물체", "OCC": "가림", "OV": "화면 밖으로 나감", "SV": "크기 변화",
    "DB": "움직이는 배경", "SC": "복잡한 경계", "AC": "겉모습 변화",
    "LRA": "오래 사라졌다 재등장", "CTC": "닮은 물체 번갈아 등장",
}


def load(split: str):
    root = _split_root(split)
    return videos_from_folders(f"lvos_v2_{split}", root / "JPEGImages", root / "Annotations",
                               ignore_value=IGNORE_VALUE)


def load_labels(split: str) -> dict:
    """[추가] 공식 속성 파일 <split>/*attribute*.json 의 videos[영상]["attributes"].
    영상 단위 → {영상: {"*": ["OCC 가림", ...]}}

    파일은 영상 zip 에 없어 따로 받는다: 공식 홈페이지 dataset 페이지 "Jsons with attributes" 의
    val_meta_attribute.json → <LVOSv2>/extracted/valid/ (서버에 넣음, 2026-10-05).
    모양: {"sets", "attributes": 13종 약자, "videos": {영상: {"attributes": ["BC", "LR", ...]}}}
    파일이 없으면 빈 결과 → LVOS 라벨 표는 "(없음)", 합친 표에서 LVOS 몫이 빠짐.
    """
    root = _split_root(split)
    files = sorted(root.glob("*attribute*.json")) if root.is_dir() else []
    if not files:
        return {}
    videos = json.loads(files[0].read_text(encoding="utf-8"))["videos"]
    return {name: {"*": [f"{a} {ATTRIBUTES[a]}" if a in ATTRIBUTES else str(a)
                         for a in info.get("attributes", [])]}
            for name, info in videos.items()}


if __name__ == "__main__":
    print_first_video(load("valid"))
