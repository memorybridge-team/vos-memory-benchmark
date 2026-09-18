"""Post-switch scores for one object's per-frame J&F curve.

Only frames whose ground truth is non-empty count (GT-visible).  Warm Target --
the large model running the whole video by itself -- is scored over the same
frames, and Retention is formed from the two at aggregation time (results.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import masked_mean

__all__ = ["HandoffScores", "post_switch_gtvis", "score_curve"]


@dataclass
class HandoffScores:
    jf_gtvis: float          # mean J&F over post-switch frames with visible GT
    jf_at_5_gtvis: float     # same, restricted to the 5 frames after the switch
    warm_jf_gtvis: float     # Warm Target's jf_gtvis on the same frames


def _window(curve: np.ndarray, switch: int, k: int | None) -> np.ndarray:
    """Frames strictly after the switch: [switch+1, switch+1+k)."""
    start = switch + 1
    stop = len(curve) if k is None else min(len(curve), start + k)
    return curve[start:stop]


def post_switch_gtvis(
    curve: np.ndarray, visible: np.ndarray, switch: int, k: int | None = None
) -> float:
    """Mean J&F over the frames in the window whose ground truth is non-empty.

    nan when the window holds no visible frame at all -- an object occluded
    across the whole window.  Those rows are dropped at aggregation time, never
    filled with 0.
    """
    return masked_mean(_window(curve, switch, k), _window(visible, switch, k))


def score_curve(
    curve: np.ndarray,
    warm: np.ndarray,
    switch: int,
    visible: np.ndarray,
) -> HandoffScores:
    """Score one object's post-switch curve.

    `visible` is a full-length boolean array, True where this object's ground
    truth is non-empty.  `warm_jf_gtvis` is carried on every row rather than
    divided into on the spot: Retention is a ratio of means, not a mean of
    ratios.
    """
    curve = np.asarray(curve, dtype=float)
    warm = np.asarray(warm, dtype=float)
    visible = np.asarray(visible, dtype=bool)
    return HandoffScores(
        jf_gtvis=post_switch_gtvis(curve, visible, switch, None),
        jf_at_5_gtvis=post_switch_gtvis(curve, visible, switch, 5),
        warm_jf_gtvis=post_switch_gtvis(warm, visible, switch, None),
    )
