"""데이터셋 이름 → 영상 목록.

    load_dataset("lvos_v2_valid")     → [Video, ...]
    load_video_list("lvos_v2_valid")  → 1_make_video_list.py 가 고정해 둔 영상·객체·전환 시점 목록
    load_labels("lvos_v2_valid")      → [추가] 데이터셋 공식 라벨 (있는 데이터셋만)
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import settings

# 데이터셋 이름 → (data/ 안의 파일, 나누기). 검증 = LVOS v2 valid, 평가 = VOST val + M3VOS + PUMaVOS
DATASETS = {
    "lvos_v2_valid": ("lvos_v2", "valid"),
    "vost_val": ("vost", "val"),
    "m3vos": ("m3vos", None),
    "pumavos": ("pumavos", None),
}


def load_dataset(name: str, *, names=None):
    module_name, split = DATASETS[name]
    module = importlib.import_module(f"evaluation.data.{module_name}")
    return module.load(split, names=names)


def load_labels(name: str) -> dict:
    """[추가] 데이터셋 공식 라벨 {영상: {객체 번호 또는 "*"(영상 전체): [라벨, ...]}}. 없으면 빈 dict."""
    module_name, split = DATASETS[name]
    module = importlib.import_module(f"evaluation.data.{module_name}")
    return module.load_labels(split) if hasattr(module, "load_labels") else {}


def labels_of(labels: dict, video: str, obj_id: int) -> list[str]:
    per_video = labels.get(video, {})
    return per_video.get("*", []) + per_video.get(obj_id, [])


def video_list_path(name: str) -> Path:
    return Path(settings.OUTPUT_ROOT) / "lists" / f"{name}.json"


def load_video_list(name: str) -> dict:
    path = video_list_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. scripts/1_make_video_list.py 를 먼저 실행하세요.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("evaluation_revision") != settings.VIDEO_LIST_REVISION
            or data.get("switch_basis") != "object"
            or data.get("switch_fractions") != list(settings.SWITCH_FRACTIONS)
            or data.get('exclude_if_pre_switch_frames_lte') != settings.MIN_TRACK_FRAMES
            or data.get('exclusion_scope') != 'all_switches_methods_runs'):
        raise ValueError(f"{path}의 평가 기준이 오래되었습니다. scripts/1_make_video_list.py를 다시 실행하세요.")
    from evaluation.switches import object_exclusion
    for entry in data['videos']:
        for obj in entry['objects']:
            reason = object_exclusion(obj)
            if reason is not None:
                raise ValueError(f"{entry['video']} 객체 {obj['object']}는 전체 평가 제외 대상입니다: {reason}. "
                                 "scripts/1_make_video_list.py를 다시 실행하세요.")
    return data
