"""비교군 전체 목록: 이름 → 함수, "main"(확정 10개) / "extra"(추가) 표시.

prepare(session, pkg, stats) → (추적 시작 프레임, 다시 본 프레임 수)
    session : 새로 연 Base+ 세션
    pkg     : 전환 순간 Small 쪽에서 챙긴 기억 상자 (model/memory.py)
    stats   : Moment-Matched 용 통계
    추적 시작 프레임이 None 이면 Base+ 가 아무것도 못 받은 것 → 전환 뒤 전부 빈 마스크.

Source-only 와 Full Replay 는 전환 시점과 무관해서 evaluate_video.py 가 따로 한 번만 돌린다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import settings
from benchmark.methods import anchors, extra_diagnostic, replay, state_copy


@dataclass(frozen=True)
class Method:
    name: str                      # 결과 파일에 쓰는 이름
    label: str                     # 표에 쓰는 이름
    role: str                      # "main" = 확정 10개, "extra" = 추가
    prepare: Callable | None = None


METHODS = [
    Method("source_only", "Source-only", "main"),
    Method("full_replay", "Full Replay", "main"),
    Method("direct_state_copy", "Direct State Copy", "main", state_copy.direct_copy),
    Method("moment_matched_copy", "Moment-Matched Copy", "main", state_copy.moment_matched_copy),
    Method("original_prompt", "Original-Prompt(s)", "main", anchors.original_prompt),
    Method("last_visible", "Last-Visible", "main", anchors.last_visible),
    Method("original_last_visible", "Original+Last-Visible", "main", anchors.original_last_visible),
    *[Method(f"original_replay_{k}", f"Original+Replay-{k}", "main", replay.make_replay(k))
      for k in settings.REPLAY_KS],
    Method("reset", "[추가] reset", "extra", extra_diagnostic.reset),
    Method("last_mask", "[추가] last_mask", "extra", extra_diagnostic.last_mask),
    Method("recent_k_only", f"[추가] recent_{settings.EXTRA_RECENT_K}_only", "extra",
           extra_diagnostic.recent_k_only),
]

BY_NAME = {m.name: m for m in METHODS}
MAIN = [m for m in METHODS if m.role == "main"]
EXTRA = [m for m in METHODS if m.role == "extra"]
