"""Per-channel statistics of a model's memory, accumulated in one streaming pass.

Why streaming: one SAM 2 memory is 64x64x64 = 262,144 floats.  Keeping the
tensors themselves for a 60-video split would run to hundreds of gigabytes, and
we do not need them -- a channel mean and standard deviation is four numbers per
channel, and they can be accumulated frame by frame and thrown away.

Why per *group* (`prompted` / `recent` / `obj_ptr`): the three carry different
things.  A prompted memory encodes a mask the model was handed; a recent memory
encodes one it produced and will keep re-reading; an obj_ptr is not a spatial
map at all.  One pooled statistic would be a blend of three distributions and
would fit none of them.

Nothing here needs paired data.  Each model's statistics are collected on its
own, from its own run; `norm_matched` only ever puts two independently-measured
summaries side by side.  That is the whole reason it can run before the paired
pipeline exists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["ChannelStats", "StatsBook", "GROUPS", "group_of"]

GROUPS = ("prompted", "recent", "obj_ptr")

EPS = 1e-6  # guards a dead channel (std 0) from turning the map into inf


def group_of(kind: str, field_name: str) -> str:
    """Which statistics group a (slot kind, tensor) pair belongs to."""
    if field_name == "obj_ptr":
        return "obj_ptr"
    return kind  # "prompted" or "recent", for maskmem


@dataclass
class ChannelStats:
    """Running mean/std over the channel axis, pooled across every other axis.

    For a (C, H, W) memory that means C statistics, each over H*W*frames values:
    the spatial layout is deliberately averaged out, because a per-position
    statistic would be fitting the dataset's camera framing, not the model's
    representation.
    """

    count: int = 0                      # values contributing per channel
    _sum: np.ndarray | None = None      # (C,)
    _sumsq: np.ndarray | None = None    # (C,)

    def update(self, arr: np.ndarray) -> None:
        """`arr` is (C,) or (C, ...); every axis after the first is pooled."""
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        else:
            a = a.reshape(a.shape[0], -1)
        if self._sum is None:
            self._sum = np.zeros(a.shape[0], dtype=np.float64)
            self._sumsq = np.zeros(a.shape[0], dtype=np.float64)
        elif self._sum.shape[0] != a.shape[0]:
            raise ValueError(
                "channel count changed mid-stream: %d then %d"
                % (self._sum.shape[0], a.shape[0])
            )
        self._sum += a.sum(axis=1)
        self._sumsq += (a * a).sum(axis=1)
        self.count += a.shape[1]

    @property
    def mean(self) -> np.ndarray:
        if not self.count:
            raise ValueError("no samples")
        return self._sum / self.count

    @property
    def std(self) -> np.ndarray:
        if not self.count:
            raise ValueError("no samples")
        var = self._sumsq / self.count - self.mean ** 2
        return np.sqrt(np.maximum(var, 0.0))

    def to_json(self) -> dict:
        return {
            "count": int(self.count),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


@dataclass
class _Frozen:
    mean: np.ndarray
    std: np.ndarray
    count: int


@dataclass
class StatsBook:
    """One model's statistics for every group, plus the provenance that makes
    them auditable: which split they came from and how much data."""

    model_id: str
    split: str = "train"
    groups: dict[str, _Frozen] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def get(self, group: str) -> _Frozen:
        if group not in self.groups:
            raise KeyError(
                "no %r statistics for %s; fit them with scripts/fit_norm_stats.py"
                % (group, self.model_id)
            )
        return self.groups[group]

    @classmethod
    def from_accumulators(
        cls, model_id: str, split: str, acc: dict[str, ChannelStats], **meta
    ) -> "StatsBook":
        return cls(
            model_id=model_id,
            split=split,
            groups={
                g: _Frozen(s.mean, np.maximum(s.std, EPS), s.count)
                for g, s in acc.items()
                if s.count
            },
            meta=meta,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "split": self.split,
                    "meta": self.meta,
                    "groups": {
                        g: {"mean": f.mean.tolist(), "std": f.std.tolist(),
                            "count": int(f.count)}
                        for g, f in self.groups.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "StatsBook":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model_id=raw["model_id"],
            split=raw.get("split", "train"),
            groups={
                g: _Frozen(
                    np.asarray(v["mean"], dtype=np.float64),
                    np.maximum(np.asarray(v["std"], dtype=np.float64), EPS),
                    int(v["count"]),
                )
                for g, v in raw["groups"].items()
            },
            meta=raw.get("meta", {}),
        )
