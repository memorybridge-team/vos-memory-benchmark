"""J, F 계산 (무시 영역 픽셀 제외). DAVIS 공식 평가 코드와 같은 정의.

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
    other_iou: float = 0.0   # [추가] 다른 객체 정답과 가장 많이 겹친 정도 (ID 뒤바뀜용)

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


def _boundary(mask: np.ndarray) -> np.ndarray:
    """마스크의 경계 픽셀 (DAVIS seg2bmap 과 같음)."""
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
    return b


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def f_score(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> float:
    pred, gt = _keep(pred, ignore), _keep(gt, ignore)
    radius = int(np.ceil(settings.BOUNDARY_THRESHOLD * np.hypot(*pred.shape)))
    pred_b, gt_b = _boundary(pred), _boundary(gt)
    n_pred, n_gt = pred_b.sum(), gt_b.sum()
    if n_pred == 0 and n_gt == 0:
        return 1.0
    if n_pred == 0 or n_gt == 0:
        return 0.0
    disk = _disk(radius)
    pred_near = cv2.dilate(pred_b.astype(np.uint8), disk).astype(bool)
    gt_near = cv2.dilate(gt_b.astype(np.uint8), disk).astype(bool)
    precision = (pred_b & gt_near).sum() / n_pred
    recall = (gt_b & pred_near).sum() / n_gt
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
