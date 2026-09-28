"""Moment-Matched Copy 용 통계: 모델마다 기억 값의 채널별 평균·표준편차.

train fit 영상을 Small, Base+ 로 각각 돌려서, 프레임마다 새로 생긴 기억 칸의
maskmem_features (64채널) 와 obj_ptr (256채널) 값을 모은다.
결과: outputs/moment_stats.npz — stats[모델][칸] = {"mean": 배열, "std": 배열}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import settings
from benchmark.model.memory import MATCHED_FIELDS

MIN_STD = 1e-6


class ChannelStats:
    """채널(1번 축)마다 합과 제곱합을 쌓아 평균·표준편차를 낸다."""

    def __init__(self):
        self.n = 0
        self.total = None
        self.total_sq = None

    def add(self, tensor) -> None:
        x = tensor.detach().float().cpu().numpy()
        x = np.moveaxis(x, 1, -1).reshape(-1, x.shape[1]).astype(np.float64)
        if self.total is None:
            self.total = np.zeros(x.shape[1])
            self.total_sq = np.zeros(x.shape[1])
        self.n += x.shape[0]
        self.total += x.sum(0)
        self.total_sq += (x ** 2).sum(0)

    def result(self) -> dict:
        mean = self.total / self.n
        var = np.maximum(self.total_sq / self.n - mean ** 2, 0.0)
        return {"mean": mean, "std": np.maximum(np.sqrt(var), MIN_STD)}


def collect(runner, items) -> dict:
    """items = [(Video, 객체 정보 dict), ...] 를 runner 로 돌려 통계를 낸다."""
    acc = {field: ChannelStats() for field in MATCHED_FIELDS}
    for i, (video, obj) in enumerate(items, 1):
        start = obj["start"]
        last = min(obj["end"], start + settings.MOMENT_STATS_MAX_FRAMES)
        session = runner.start(video)
        session.add_prompt(start, video.object_mask(start, obj["object"]))
        for out in session.track(start, last):
            entry = session.memory_of(out.frame)
            if entry is not None:
                for field in MATCHED_FIELDS:
                    acc[field].add(entry[field])
        session.close()
        print(f"  [{runner.name}] {i}/{len(items)} {video.dataset}/{video.name}")
    return {field: a.result() for field, a in acc.items()}


def stats_path() -> Path:
    return Path(settings.OUTPUT_ROOT) / "moment_stats.npz"


def save(stats: dict) -> Path:
    path = stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {f"{model}.{field}.{k}": v
            for model, fields in stats.items()
            for field, values in fields.items()
            for k, v in values.items()}
    np.savez(path, **flat)
    return path


def load() -> dict:
    path = stats_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. scripts/2_fit_moment_stats.py 를 먼저 실행하세요.")
    stats: dict = {}
    with np.load(path) as data:
        for key in data.files:
            model, field, k = key.split(".")
            stats.setdefault(model, {}).setdefault(field, {})[k] = data[key]
    return stats
