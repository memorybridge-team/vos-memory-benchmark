"""데이터셋 이름 → 영상 목록.

    load_dataset("lvos_v2_valid")     → [Video, ...]
    load_video_list("lvos_v2_valid")  → 1_make_video_list.py 가 고정해 둔 영상·객체·전환 시점 목록
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import settings

# 데이터셋 이름 → (data/ 안의 파일, 나누기)
DATASETS = {
    "mosev2_train": ("mosev2", "train"),
    "lvos_v2_train": ("lvos_v2", "train"),
    "mosev2_valid": ("mosev2", "valid"),
    "lvos_v2_valid": ("lvos_v2", "valid"),
    "vost_val": ("vost", "val"),
    "m3vos": ("m3vos", None),
    "pumavos": ("pumavos", None),
}


def load_dataset(name: str):
    module_name, split = DATASETS[name]
    module = importlib.import_module(f"benchmark.data.{module_name}")
    return module.load(split)


def video_list_path(name: str) -> Path:
    return Path(settings.OUTPUT_ROOT) / "lists" / f"{name}.json"


def load_video_list(name: str) -> dict:
    path = video_list_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. scripts/1_make_video_list.py 를 먼저 실행하세요.")
    return json.loads(path.read_text(encoding="utf-8"))
