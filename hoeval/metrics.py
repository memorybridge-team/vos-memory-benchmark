"""DAVIS-style region (J) and boundary (F) similarity.

Reimplementation of the official DAVIS 2017 evaluation primitives so that the
handoff study does not depend on the davis2017-evaluation package (which pins
old numpy/skimage).  Verified against the reference implementation's behaviour
on the degenerate cases that matter here: empty prediction, empty ground truth,
and both empty (an object that is fully occluded on a frame).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

__all__ = ["region_similarity", "boundary_similarity", "jf", "seg2bmap",
           "gt_visible", "masked_mean"]


def region_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """J = intersection-over-union of two binary masks.

    Both empty counts as a perfect match: the object is absent and the model
    correctly predicted nothing.  This is why the headline numbers are GT-visible:
    averaged in, these free 1.0s reward a model that has lost the object.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.count_nonzero(pred | gt)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pred & gt) / union)


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Binary boundary map of a mask (1px wide, no boundary on the image edge)."""
    seg = seg.astype(bool)
    h, w = seg.shape
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, : w - 1] = seg[:, 1:]
    s[: h - 1, :] = seg[1:, :]
    se[: h - 1, : w - 1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    return b


def _disk(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def boundary_similarity(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> float:
    """F = harmonic mean of boundary precision and recall under a tolerance.

    `bound_th` is a fraction of the image diagonal (DAVIS default 0.008), i.e.
    a predicted boundary pixel counts as correct if a ground-truth boundary
    pixel lies within that many pixels.
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(pred.shape))
    se = _disk(bound_pix)

    fg_b = seg2bmap(pred)
    gt_b = seg2bmap(gt)
    fg_dil = binary_dilation(fg_b, se)
    gt_dil = binary_dilation(gt_b, se)

    gt_match = gt_b & fg_dil
    fg_match = fg_b & gt_dil

    n_fg = np.count_nonzero(fg_b)
    n_gt = np.count_nonzero(gt_b)

    if n_fg == 0 and n_gt == 0:
        return 1.0
    if n_fg == 0 or n_gt == 0:
        return 0.0

    precision = np.count_nonzero(fg_match) / n_fg
    recall = np.count_nonzero(gt_match) / n_gt
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def jf(pred: np.ndarray, gt: np.ndarray, bound_th: float = 0.008) -> tuple[float, float, float]:
    """Return (J, F, J&F) for one binary mask pair."""
    j = region_similarity(pred, gt)
    f = boundary_similarity(pred, gt, bound_th)
    return j, f, 0.5 * (j + f)


# --- GT-visible filtering ---------------------------------------------------
#
# `region_similarity` scores both-empty as 1.0, which is right per frame and
# wrong in aggregate: on a video where the object is hidden for half its length,
# a model that predicts nothing at all collects a free 1.0 on every hidden
# frame.  Averaged over the video that reads as competence.  So every headline
# number in this study is computed over the frames where the ground truth is
# non-empty, and the unfiltered average is kept beside it only as a diagnostic.


def gt_visible(gt: np.ndarray) -> bool:
    """True when the object actually appears in this frame's ground truth."""
    return bool(np.any(gt))


def masked_mean(values: np.ndarray, keep: np.ndarray) -> float:
    """Mean of `values` over `keep`, or nan when nothing is kept.

    nan rather than 0 on purpose: an empty window means "not measured here",
    and the aggregation layer must drop it, not average it in as a failure.
    """
    values = np.asarray(values, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    if values.size == 0 or keep.size == 0:
        return float("nan")
    n = min(values.size, keep.size)
    sel = values[:n][keep[:n]]
    sel = sel[~np.isnan(sel)]
    return float(sel.mean()) if sel.size else float("nan")
