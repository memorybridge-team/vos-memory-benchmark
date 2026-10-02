"""Small → Base+ 로 넘기는 기억 상자.

전환 순간 Small 쪽에서 챙겨 두는 것 전부를 한 상자(HandoffPackage)에 담는다.
비교군마다 이 상자에서 필요한 것만 꺼내 Base+ 에 넘긴다.

    small_memory    SAM2 내부 기억 칸 (Direct State Copy / Moment-Matched Copy 가 씀)
    prompt_*        처음 받은 정답 마스크 (Original-Prompt, Replay 가 씀)
    last_visible    Small이 마지막으로 "보인다"고 한 프레임과 그 마스크 (Last-Visible 이 씀)
    recent_masks    Small이 최근 몇 프레임에 낸 마스크 ([추가] recent_k_only 가 씀)

정답은 처음 프롬프트 한 번만 들어간다. 나머지는 모두 Small이 스스로 낸 것.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import settings

MATCHED_FIELDS = ("maskmem_features", "obj_ptr")   # Moment-Matched 가 맞추는 칸


@dataclass
class HandoffPackage:
    switch_frame: int                         # Small이 여기까지 봤다 (Base+ 는 다음 프레임부터)
    prompt_frame: int
    prompt_mask: np.ndarray
    small_memory: dict                        # {프레임: 기억 칸} — sam2_runner.export_memory()
    last_visible: tuple[int, np.ndarray] | None
    recent_masks: dict[int, np.ndarray]       # {프레임: Small 마스크}

    def last_visible_or_prompt(self) -> tuple[int, np.ndarray]:
        """Small이 한 번도 '보인다'고 한 적이 없으면 처음 프롬프트를 쓴다."""
        if self.last_visible is not None:
            return self.last_visible
        return self.prompt_frame, self.prompt_mask


def _per_channel(x: torch.Tensor, scale: np.ndarray, shift: np.ndarray) -> torch.Tensor:
    """채널(1번 축)마다 x * scale + shift."""
    shape = [1, -1] + [1] * (x.dim() - 2)
    s = torch.as_tensor(scale, dtype=torch.float32, device=x.device).view(shape)
    b = torch.as_tensor(shift, dtype=torch.float32, device=x.device).view(shape)
    return (x.float() * s + b).to(x.dtype)


def match_moments(entries: dict, stats: dict) -> dict:
    """Small 기억 값을 Base+ 기억 값의 평균·표준편차에 맞춘다 (채널마다).

        x' = (x - Small평균) / Small표준편차 × Base+표준편차 + Base+평균

    stats 는 moment_stats.load() 결과: stats[모델][칸] = {"mean", "std"}.
    """
    src, dst = stats[settings.SOURCE_MODEL], stats[settings.TARGET_MODEL]
    matched = {}
    for frame, entry in entries.items():
        new = dict(entry)
        for field in MATCHED_FIELDS:
            scale = dst[field]["std"] / src[field]["std"]
            shift = dst[field]["mean"] - src[field]["mean"] * scale
            new[field] = _per_channel(entry[field], scale, shift)
        matched[frame] = new
    return matched
