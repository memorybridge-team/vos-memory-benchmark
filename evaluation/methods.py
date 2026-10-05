"""평가하는 방법 전체 = 비교군 (baseline/) + 본 모델 (translator/). 2_evaluate.py --methods 로 일부만 고른다.

    --methods all          전부 (기본)
    --methods baselines    비교군만
    --methods model        본 모델만
    --methods <이름> ...    방법 이름 (예: direct_state_copy translator)
Full Replay · Source-only 는 회복률·격차 회복률의 기준이라 무엇을 고르든 같이 낸다.
"""

from __future__ import annotations

from baseline import BASELINES
from translator import MODEL

METHODS = BASELINES + [MODEL]
BY_NAME = {m.name: m for m in METHODS}
GROUPS = {"all": METHODS, "baselines": BASELINES, "model": [MODEL]}
ALWAYS = ("full_replay", "source_only")


def select(names: list[str]) -> list:
    chosen = set(ALWAYS)
    for name in names:
        chosen |= {m.name for m in GROUPS[name]} if name in GROUPS else {BY_NAME[name].name}
    return [m for m in METHODS if m.name in chosen]
