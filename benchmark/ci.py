"""영상 단위 신뢰구간 (부트스트랩).

영상들을 복원 추출로 다시 뽑아 평균을 여러 번 내고, 그 가운데 95% 가 들어가는 범위를 쓴다.
영상 단위로 뽑는 이유: 한 영상 안의 객체·전환 시점들은 서로 닮아서 따로 세면 범위가 너무 좁아진다.
"""

from __future__ import annotations

import numpy as np

import settings


def bootstrap(values) -> tuple[float, float, float] | None:
    """(평균, 아래, 위). 값이 없으면 None."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return None
    rng = np.random.default_rng(settings.BOOTSTRAP_SEED)
    picks = rng.integers(0, len(values), size=(settings.BOOTSTRAP_SAMPLES, len(values)))
    means = values[picks].mean(axis=1)
    tail = (1 - settings.CI_LEVEL) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return float(values.mean()), float(lo), float(hi)
