"""Work on numpy or torch tensors without importing torch.

Translators are fitted and unit-tested on a laptop with numpy only, then run on
the server against real torch tensors.  Everything below round-trips through
numpy for the arithmetic and restores the caller's original container, dtype
and device, so the same translator code serves both.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["to_numpy", "like"]


def _is_torch(x: Any) -> bool:
    return hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy")


def to_numpy(x: Any) -> np.ndarray:
    if _is_torch(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def like(ref: Any, arr: np.ndarray) -> Any:
    """`arr` as the same kind of object as `ref` (dtype and device included)."""
    if _is_torch(ref):
        import torch  # noqa: PLC0415  -- only on the path that already has it

        return torch.as_tensor(arr, dtype=ref.dtype, device=ref.device)
    return np.asarray(arr, dtype=getattr(ref, "dtype", None))
