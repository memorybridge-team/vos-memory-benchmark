"""평가하는 방법 전체 = 비교군 (baseline/) + 본 모델 (translator/). 2_evaluate.py 가 한 번에 모두 돌린다."""

from baseline import BASELINES
from translator import MODEL

METHODS = BASELINES + [MODEL]
