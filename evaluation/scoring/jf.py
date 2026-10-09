"""J, F 계산 (무시 영역 픽셀 제외). F의 허용 거리는 이미지 대각선 × 0.008.

J = 겹친 넓이 / 합친 넓이 (IoU). 둘 다 비어 있으면 1.
F = 경계선끼리 얼마나 가까운가. 대각선 × BOUNDARY_THRESHOLD 픽셀 안이면 맞은 것으로 본다.
J&F = (J + F) / 2
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import settings


@dataclass
class FrameScore:
    j: float
    f: float
    gt_visible: bool         # 이 프레임 정답에 객체가 보이는가

    @property
    def jf(self) -> float:
        return (self.j + self.f) / 2


def _keep(mask: np.ndarray, ignore: np.ndarray | None) -> np.ndarray:
    return mask if ignore is None else mask & ~ignore


def j_score(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> float:
    pred, gt = _keep(pred, ignore), _keep(gt, ignore)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def _boundary(mask: np.ndarray, ignore: np.ndarray | None = None) -> np.ndarray:
    """DAVIS seg2bmap 경계. 무시 픽셀이 포함된 비교는 채점에서 제외한다."""
    e = np.zeros_like(mask)
    s = np.zeros_like(mask)
    se = np.zeros_like(mask)
    e[:, :-1] = mask[:, 1:]
    s[:-1, :] = mask[1:, :]
    se[:-1, :-1] = mask[1:, 1:]
    b = (mask ^ e) | (mask ^ s) | (mask ^ se)
    b[-1, :] = mask[-1, :] ^ e[-1, :]
    b[:, -1] = mask[:, -1] ^ s[:, -1]
    b[-1, -1] = False
    if ignore is not None:
        valid = ~ignore
        comparisons = valid.copy()
        comparisons[:-1, :-1] &= (valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:])
        comparisons[-1, :-1] &= valid[-1, 1:]
        comparisons[:-1, -1] &= valid[1:, -1]
        b &= comparisons
    return b


def f_score(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> float:
    pred, gt = _keep(pred, ignore), _keep(gt, ignore)
    threshold = settings.BOUNDARY_THRESHOLD * np.hypot(*pred.shape)
    pred_b, gt_b = _boundary(pred, ignore), _boundary(gt, ignore)
    n_pred, n_gt = pred_b.sum(), gt_b.sum()
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    # 허용 거리를 올림하지 않고, 각 테두리 점의 정확한 유클리드 거리를 비교한다.
    to_gt = cv2.distanceTransform((~gt_b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    to_pred = cv2.distanceTransform((~pred_b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    precision = (to_gt[pred_b] <= threshold).sum() / n_pred
    recall = (to_pred[gt_b] <= threshold).sum() / n_gt
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
