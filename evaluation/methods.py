"""평가하는 방법 전체 = 비교군 (baseline/) + 본 모델 (translator/). 2_evaluate.py 가 한 번에 모두 돌린다."""

import settings
from baseline import BASELINES
from translator import MODEL

METHODS = BASELINES + [MODEL]       # 표에 나오는 전부


def to_run() -> list:
    """2_evaluate.py 가 돌리는 것. RUN_EXTRA 가 꺼져 있으면 진단 비교군(role "extra")은 뺀다."""
    return METHODS if settings.RUN_EXTRA else [m for m in METHODS if m.role != "extra"]
