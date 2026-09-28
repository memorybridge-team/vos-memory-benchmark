"""가짜 데이터셋: 색깔 상자 몇 개가 움직이는 작은 영상들.

객체 1  빨강, 오른쪽으로 이동
객체 2  초록, 아래로 이동, 12~17 프레임 동안 사라졌다 다시 나타남 (전환 B 용)
객체 3  파랑, 5 프레임부터 나타남 (객체가 중간에 시작하는 경우)
무시 영역 오른쪽 아래 구석 (값 255, 회색)
"""

from pathlib import Path

import numpy as np
from PIL import Image

H, W, N = 36, 48, 30
COLORS = {1: (200, 30, 30), 2: (30, 200, 30), 3: (30, 30, 200)}
PALETTE = [0, 0, 0, 200, 30, 30, 30, 200, 30, 30, 30, 200] + [128] * (256 * 3 - 12)


def boxes(t: int) -> dict:
    """프레임 t 에서 객체별 (y0, x0, h, w). 안 보이면 빠짐."""
    out = {1: (4, 2 + t, 8, 10)}
    if not 12 <= t < 18:
        out[2] = (14 + t // 3, 20, 8, 10)
    if t >= 5:
        out[3] = (26, 2, 6, 8)
    return out


def draw(t: int, ignore: bool):
    img = np.zeros((H, W, 3), dtype=np.uint8)
    labels = np.zeros((H, W), dtype=np.uint8)
    for obj_id, (y, x, h, w) in boxes(t).items():
        img[y:y + h, x:x + w] = COLORS[obj_id]
        labels[y:y + h, x:x + w] = obj_id
    if ignore:
        img[H - 3:, W - 3:] = 128
        labels[H - 3:, W - 3:] = 255
    return img, labels


def make_video(frames_root: Path, masks_root: Path, name: str,
               first_frame_only=False, ignore=False, shift=0) -> None:
    (frames_root / name).mkdir(parents=True, exist_ok=True)
    (masks_root / name).mkdir(parents=True, exist_ok=True)
    for t in range(N):
        img, labels = draw(t + shift, ignore)
        Image.fromarray(img).save(frames_root / name / f"{t:05d}.png")
        if first_frame_only and t > 0:
            continue
        png = Image.fromarray(labels, mode="P")
        png.putpalette(PALETTE)
        png.save(masks_root / name / f"{t:05d}.png")


def make_all(data_root: Path, folders: dict) -> None:
    """settings.DATA_FOLDERS 와 data/*.py 가 기대하는 폴더 모양으로 만든다."""
    def split_layout(key, split, names, **kw):
        root = data_root / folders[key] / split
        for i, name in enumerate(names):
            make_video(root / "JPEGImages", root / "Annotations", name, shift=i % 3, **kw)

    split_layout("mosev2", "train", [f"mtrain_{i:02d}" for i in range(6)])
    split_layout("mosev2", "valid", ["mval_00", "mval_01"], first_frame_only=True)
    split_layout("lvos_v2", "train", [f"ltrain_{i:02d}" for i in range(10)])
    split_layout("lvos_v2", "valid", ["lval_00", "lval_01", "lval_02"])

    vost = data_root / folders["vost"]
    for name in ("vost_00", "vost_01"):
        make_video(vost / "JPEGImages", vost / "Annotations", name, ignore=True)
    (vost / "ImageSets").mkdir(parents=True, exist_ok=True)
    (vost / "ImageSets" / "val.txt").write_text("vost_00\nvost_01\n")

    for key, ignore in (("m3vos", True), ("pumavos", False)):
        root = data_root / folders[key]
        make_video(root / "JPEGImages", root / "Annotations", f"{key}_00", ignore=ignore)
