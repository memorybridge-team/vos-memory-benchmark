"""A tiny DAVIS-layout dataset of moving squares, generated on the fly.

Exists so the harness can be exercised end to end with no GPU, no checkpoints
and no dataset download.  It is a test fixture, not a benchmark.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * (256 * 3 - 12)


def make_video(root: Path, name: str, n_frames: int = 40, size: int = 128, n_obj: int = 2):
    img_dir = root / "JPEGImages" / name
    ann_dir = root / "Annotations" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    for t in range(n_frames):
        ids = np.zeros((size, size), dtype=np.uint8)
        rgb = np.zeros((size, size, 3), dtype=np.uint8)
        for o in range(1, n_obj + 1):
            # objects drift diagonally at different speeds and cross each other
            cx = 20 + (t * (2 + o)) % (size - 45)
            cy = 20 + (t * 2 * o) % (size - 45)
            ids[cy : cy + 22, cx : cx + 22] = o
            rgb[cy : cy + 22, cx : cx + 22] = (60 * o, 200 - 60 * o, 120)
        Image.fromarray(rgb).save(img_dir / ("%05d.jpg" % t), quality=95)
        png = Image.fromarray(ids, mode="P")
        png.putpalette(PALETTE)
        png.save(ann_dir / ("%05d.png" % t))


def make_dataset(root: Path, videos=("square_a", "square_b"), n_frames: int = 40):
    for v in videos:
        make_video(root, v, n_frames=n_frames)
    return root
