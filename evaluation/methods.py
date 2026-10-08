"""평가하는 방법 전체 = 비교군 (baseline/) + 본 모델 (translator/). 2_evaluate.py 가 한 번에 모두 돌린다."""

from baseline import BASELINES
from translator import MODEL

METHODS = BASELINES + [MODEL]       # 표에 나오는 전부


def to_run() -> list:
    """비교군 5개 + 본 모델. 모든 요청 지표를 항상 계산한다."""
    return METHODS
