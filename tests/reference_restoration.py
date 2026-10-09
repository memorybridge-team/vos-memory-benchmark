"""b05a637의 변경 전 R² 기준. Git 이력 없이 캐시 동등성 테스트를 실행한다."""

import math
import torch

def tensor_score(native, prepared):
    """한 필드의 tensor 원소를 펼친 R²과 재집계용 충분통계량. 계산은 CPU float64."""
    result = {'r2': None, 'sse': None, 'sst': None, 'native_mean': None,
              'n_elements': 0,
              'native_shape': list(native.shape) if native is not None else None,
              'target_shape': list(prepared.shape) if prepared is not None else None}
    if native is None or prepared is None:
        return {**result, 'status': 'missing_native_tensor' if native is None else 'missing_target_tensor'}
    if tuple(native.shape) != tuple(prepared.shape):
        return {**result, 'status': 'shape_mismatch'}
    y = native.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
    pred = prepared.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
    if not y.numel():
        return {**result, 'status': 'empty_tensor'}
    if not bool(torch.isfinite(y).all() and torch.isfinite(pred).all()):
        return {**result, 'status': 'nonfinite_tensor'}
    center = float(y.mean())
    sst = float(((y - center) ** 2).sum())
    sse = float(((pred - y) ** 2).sum())
    if not all(math.isfinite(v) for v in (center, sst, sse)):
        return {**result, 'status': 'nonfinite_statistics'}
    result.update(sse=sse, sst=sst, native_mean=center, n_elements=y.numel())
    if y.numel() < 2:
        return {**result, 'status': 'insufficient_elements'}
    if sst == 0:
        return {**result, 'status': 'zero_native_variance'}
    r2 = 1 - sse / sst
    if not math.isfinite(r2):
        return {**result, 'status': 'nonfinite_r2'}
    return {**result, 'r2': r2, 'status': 'ok'}

