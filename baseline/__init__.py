"""비교군: 전환 때 Base+ 에 무엇을 넘기나. 비교군 5개를 등록한다. 본 모델은 translator/ 에서 별도로 등록한다.

prepare(session, pkg) → 추적 시작 프레임
    session : 새로 연 Base+ 세션
    pkg     : 전환 순간 Small 쪽에서 챙긴 기억 상자 (baseline/handoff.py)
    추적 시작 프레임이 None 이면 Base+ 가 아무것도 못 받은 것 → 전환 뒤 전부 빈 마스크.

Source-only 와 Full Replay 는 전환 시점과 무관해서 evaluate_video.py 가 따로 한 번만 돌린다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import settings
from baseline import anchors, replay, state_copy


@dataclass(frozen=True)
class Baseline:
    name: str                      # 결과 파일에 쓰는 이름
    label: str                     # 표에 쓰는 이름
    role: str                      # "main" = 비교군, "model" = 본 모델
    prepare: Callable | None = None
    revision: int = 1              # 실행 정의가 바뀌면 올린다 (예전 결과 재사용 방지)


BASELINES = [
    Baseline("source_only", "Source-only", "main"),
    Baseline("full_replay", "Full Replay (Base+ Native)", "main"),
    Baseline("direct_state_copy", "Direct State Copy", "main", state_copy.direct_copy),
    Baseline("original_last_visible", "Original + Last-Visible", "main",
             anchors.original_last_visible, revision=2),
    Baseline(f"original_replay_{settings.REPLAY_FRAMES}",
             f"Original-Prompt(s)+Replay-{settings.REPLAY_FRAMES}", "main",
             replay.make_replay(settings.REPLAY_FRAMES)),
]

MAIN = BASELINES
