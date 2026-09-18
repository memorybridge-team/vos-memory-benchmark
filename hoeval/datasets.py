"""DAVIS 2017 loader.

    <DAVIS>/JPEGImages/480p/<video>/00000.jpg ...
    <DAVIS>/Annotations/480p/<video>/00000.png ...  palette PNG, px = obj id
    <DAVIS>/ImageSets/2017/{train,val}.txt

`davis2017(root, split)` builds the loader from the unpacked trainval root.
Frame 0's annotation is the only prompt; every other annotation is ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["VideoSpec", "DavisLayout", "GtCache", "davis2017"]


@dataclass(frozen=True)
class VideoSpec:
    name: str
    frames: tuple[Path, ...]
    annotations: tuple[Path, ...]
    obj_ids: tuple[int, ...]

    @property
    def num_frames(self) -> int:
        return len(self.frames)


def _load_ids(png: Path) -> np.ndarray:
    """Palette PNG -> integer id map. 0 is background."""
    return np.array(Image.open(png).convert("P"), dtype=np.uint8)


class DavisLayout:
    def __init__(
        self,
        root: str | Path,
        name: str,
        split_file: str | Path | None = None,
        subdir: str | None = None,
    ):
        self.root = Path(root)
        self.name = name
        self.subdir = subdir
        self.images = self.root / "JPEGImages"
        self.labels = self.root / "Annotations"
        if subdir:
            self.images = self.images / subdir
            self.labels = self.labels / subdir
        if not self.images.is_dir():
            raise FileNotFoundError(
                "no %s -- for DAVIS 2017 pass subdir='480p'" % self.images
            )
        if split_file is not None:
            wanted = [
                ln.strip()
                for ln in Path(split_file).read_text(encoding="utf-8").splitlines()
                if ln.strip()
            ]
        else:
            wanted = sorted(p.name for p in self.images.iterdir() if p.is_dir())
        self.video_names = tuple(wanted)

    def video(self, name: str) -> VideoSpec:
        frames = tuple(sorted((self.images / name).glob("*.jpg")))
        anns = tuple(sorted((self.labels / name).glob("*.png")))
        if not frames:
            raise FileNotFoundError("no frames for video %r" % name)
        ids = np.unique(_load_ids(anns[0])) if anns else np.array([0])
        obj_ids = tuple(int(i) for i in ids if i != 0)
        return VideoSpec(name, frames, anns, obj_ids)

    def videos(self):
        for n in self.video_names:
            yield self.video(n)

    def video_lengths(self) -> dict[str, int]:
        return {n: len(list((self.images / n).glob("*.jpg"))) for n in self.video_names}

    @staticmethod
    def gt_masks(spec: VideoSpec, frame_idx: int) -> dict[int, np.ndarray]:
        """Ground-truth binary mask per object id for one frame.

        Objects absent from the frame yield an all-false mask rather than being
        omitted, so occlusion is scored instead of skipped.
        """
        ids = _load_ids(spec.annotations[frame_idx])
        return {oid: (ids == oid) for oid in spec.obj_ids}


def davis2017(root: str | Path, split: str = "val") -> DavisLayout:
    """DAVIS 2017 at 480p, restricted to the official `split` ("train" or "val")."""
    root = Path(root)
    return DavisLayout(root, "davis2017_%s" % split,
                       split_file=root / "ImageSets" / "2017" / ("%s.txt" % split),
                       subdir="480p")


class GtCache:
    """Ground truth for one video, decoded once.

    Every method in a switch point is scored against the same annotations, and
    a DAVIS video is ~69 palette PNGs.  Decoding them per method per object
    turned out to dominate the harness on the synthetic set, so the runner holds
    one of these per video and drops it when it moves on.
    """

    def __init__(self, spec: VideoSpec):
        self.spec = spec
        self._ids: dict[int, np.ndarray] = {}

    def ids(self, frame_idx: int) -> np.ndarray:
        cached = self._ids.get(frame_idx)
        if cached is None:
            cached = _load_ids(self.spec.annotations[frame_idx])
            self._ids[frame_idx] = cached
        return cached

    def mask(self, frame_idx: int, obj_id: int) -> np.ndarray:
        return self.ids(frame_idx) == obj_id

    def masks(self, frame_idx: int) -> dict[int, np.ndarray]:
        ids = self.ids(frame_idx)
        return {oid: (ids == oid) for oid in self.spec.obj_ids}

    def visible(self, frame_idx: int, obj_id: int) -> bool:
        return bool(np.any(self.ids(frame_idx) == obj_id))
