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

import json

from benchmark.data.common import dataset_root, print_first_video, videos_from_folders

IGNORE_VALUE = None
SPLIT_FOLDERS = {"train": ("train",), "valid": ("val", "valid")}   # 우리 이름 → 실제 폴더 이름 후보
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
                               ignore_value=IGNORE_VALUE, has_full_gt=True)


def load_labels(split: str) -> dict:
    """[추가] 공식 속성 파일 <split>/*attribute*.json 의 videos[영상]["attributes"].
    영상 단위 → {영상: {"*": ["OCC 가림", ...]}}

    ※ 미확인 (2026-09-29): 이 파일이 실제로 있는지 아직 모른다.
      - 있다고 볼 근거: LVOS 논문 Table II (영상마다 속성 13종), README 의 x_meta_attribute.json 모양 설명
      - 없을 수 있는 근거: README 가 안내하는 공식 meta 다운로드 폴더에는 train/valid/test_meta.json 뿐
        (이 파일들은 객체 등장 구간만 담음). 영상 zip(train.zip, valid.zip) 안에 있는지는 못 봄
      - LVOS v2 를 서버에 받은 뒤 확인: find <LVOSv2 폴더> -iname "*attribute*"
      - 없으면 여기서 빈 결과 → LVOS 공식 라벨 표는 "(라벨 없음)", 합친 표에서 LVOS 몫이 빠짐
      - 있는데 모양이 README 와 다르면 아래 읽는 부분을 고칠 것
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
    print_first_video(load("train"))
    print_first_video(load("valid"))
