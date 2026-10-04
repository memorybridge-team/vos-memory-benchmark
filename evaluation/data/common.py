"""영상 하나 = 프레임 목록 + 정답 PNG + 무시값.

모든 데이터셋은 결국 아래 Video 하나로 바뀐다. 데이터셋마다 다른 점
(폴더 모양, 무시 영역 값)은 data/<데이터셋>.py 가 정해서 넘긴다. 모두 모든 프레임에 정답이 있다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

import settings

FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class Video:
    dataset: str                  # 예: "lvos_v2_valid"
    name: str                     # 영상 폴더 이름
    frame_paths: list[Path]       # 프레임 번호 순서대로
    mask_paths: dict[int, Path]   # 프레임 번호 → 정답 PNG (정답이 있는 프레임만)
    ignore_value: int | None      # 정답 PNG에서 "채점하지 않는 영역"의 값 (없으면 None)

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)

    @cached_property
    def size(self) -> tuple[int, int]:
        """(높이, 너비)"""
        with Image.open(self.frame_paths[0]) as im:
            return im.height, im.width

    def read_labels(self, frame: int):
        """정답 PNG → (객체 번호 지도, 무시 영역). 정답이 없는 프레임이면 None.

        무시 영역 픽셀은 객체 번호 지도에서 0(배경)으로 바꿔 둔다.
        """
        path = self.mask_paths.get(frame)
        if path is None:
            return None
        labels = read_label_png(path)
        ignore = None
        if self.ignore_value is not None:
            ignore = labels == self.ignore_value
            labels = np.where(ignore, 0, labels)
        return labels, ignore

    def object_mask(self, frame: int, obj_id: int) -> np.ndarray:
        labels, _ = self.read_labels(frame)
        return labels == obj_id


def read_label_png(path: Path) -> np.ndarray:
    """정답 PNG 하나를 객체 번호 지도(정수)로 읽는다."""
    with Image.open(path) as im:
        if im.mode in ("P", "L", "I", "I;16"):
            return np.array(im)
        if im.mode == "1":
            return np.array(im).astype(np.uint8)
        # 색으로 객체를 구분하는 PNG: 색 하나 = 번호 하나
        rgb = np.array(im.convert("RGB")).astype(np.int64)
        return rgb[..., 0] * 65536 + rgb[..., 1] * 256 + rgb[..., 2]


def object_ids(labels: np.ndarray) -> list[int]:
    """번호 지도에 있는 객체 번호들 (0 = 배경 제외)."""
    if labels.dtype == np.uint8:
        ids = np.flatnonzero(np.bincount(labels.ravel(), minlength=256))
    else:
        ids = np.unique(labels)
    return [int(i) for i in ids if i != 0]


def dataset_root(key: str) -> Path:
    return Path(settings.DATA_ROOT) / settings.DATA_FOLDERS[key]


def read_names(list_file: Path) -> list[str]:
    """ImageSets/val.txt 같은 영상 이름 목록 파일."""
    return [line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]


def _natural_key(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.stem)]


def videos_from_folders(dataset: str, frames_root: Path, masks_root: Path, *,
                        names: list[str] | None = None,
                        ignore_value: int | None = None) -> list[Video]:
    """frames_root/<영상>/<프레임>.jpg 와 masks_root/<영상>/<프레임>.png 구조를 읽는다.

    프레임과 정답은 파일 이름(확장자 뺀 것)이 같으면 짝이 된다.
    """
    frames_root, masks_root = Path(frames_root), Path(masks_root)
    if names is None:
        names = sorted(d.name for d in frames_root.iterdir() if d.is_dir())
    videos = []
    for name in names:
        frames = sorted((p for p in (frames_root / name).iterdir()
                         if p.suffix.lower() in FRAME_EXTENSIONS), key=_natural_key)
        index = {p.stem: i for i, p in enumerate(frames)}
        masks = {}
        mask_dir = masks_root / name
        if mask_dir.is_dir():
            for p in mask_dir.glob("*.png"):
                if p.stem in index:
                    masks[index[p.stem]] = p
        videos.append(Video(dataset, name, frames, masks, ignore_value))
    return videos


def print_first_video(videos: list[Video]) -> None:
    """데이터가 제대로 읽히는지 눈으로 확인: 첫 영상의 프레임 수·객체 수·무시 영역 값."""
    if not videos:
        print("영상이 없습니다. settings.DATA_ROOT / DATA_FOLDERS 를 확인하세요.")
        return
    v = videos[0]
    first_gt = min(v.mask_paths) if v.mask_paths else None
    ids = object_ids(v.read_labels(first_gt)[0]) if first_gt is not None else []
    print(f"[{v.dataset}] 영상 {len(videos)}개 중 첫 영상 '{v.name}'")
    print(f"  프레임 수        : {v.num_frames}  (크기 {v.size[0]}x{v.size[1]})")
    print(f"  정답 있는 프레임 : {len(v.mask_paths)}")
    print(f"  첫 정답 프레임   : {first_gt}, 그 프레임 객체 {len(ids)}개 {ids}")
    print(f"  무시 영역 값     : {v.ignore_value}")
