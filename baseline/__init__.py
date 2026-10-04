"""비교군: 전환 때 Base+ 에 무엇을 넘기나. 전체 목록 = 이름 → 함수, "main"(확정 9개) / "extra"(추가) 표시.

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
from baseline import anchors, extra_diagnostic, replay, state_copy


@dataclass(frozen=True)
class Baseline:
    name: str                      # 결과 파일에 쓰는 이름
    label: str                     # 표에 쓰는 이름
    role: str                      # "main" = 확정 9개, "extra" = 추가
    prepare: Callable | None = None


BASELINES = [
    Baseline("source_only", "Source-only", "main"),
    Baseline("full_replay", "Full Replay", "main"),
    Baseline("direct_state_copy", "Direct State Copy", "main", state_copy.direct_copy),
    Baseline("original_prompt", "Original-Prompt(s)", "main", anchors.original_prompt),
    Baseline("last_visible", "Last-Visible", "main", anchors.last_visible),
    Baseline("original_last_visible", "Original+Last-Visible", "main", anchors.original_last_visible),
    *[Baseline(f"original_replay_{k}", f"Original+Replay-{k}", "main", replay.make_replay(k))
      for k in settings.REPLAY_KS],
    Baseline("reset", "[extra] reset", "extra", extra_diagnostic.reset),
    Baseline("recent_k_only", f"[extra] recent_{settings.EXTRA_RECENT_K}_only", "extra",
           extra_diagnostic.recent_k_only),
]

BY_NAME = {m.name: m for m in BASELINES}
MAIN = [m for m in BASELINES if m.role == "main"]
EXTRA = [m for m in BASELINES if m.role == "extra"]
