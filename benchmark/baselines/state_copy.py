"""Small 기억 칸을 그대로 Base+ 에 넣는 두 비교군. 다시 보는 프레임 0장.

Direct State Copy   : 그대로 복사.
Moment-Matched Copy : 값의 평균·표준편차를 Base+ 쪽에 맞춘 뒤 복사 (통계는 train fit 에서).

prepare 규칙 (모든 비교군 공통): Base+ 세션을 준비하고 (추적 시작 프레임, 다시 본 프레임 수) 를 돌려준다.
"""

from benchmark.model.memory import match_moments


def direct_copy(session, pkg, stats):
    session.load_memory(pkg.small_memory)
    return pkg.switch_frame + 1, 0


def moment_matched_copy(session, pkg, stats):
    session.load_memory(match_moments(pkg.small_memory, stats))
    return pkg.switch_frame + 1, 0
