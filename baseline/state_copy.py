"""Direct State Copy: Small 기억 칸을 그대로 Base+ 에 넣는다.

prepare 규칙 (모든 비교군 공통): Base+ 세션을 준비하고 추적 시작 프레임을 돌려준다.
"""


def direct_copy(session, pkg):
    session.load_memory(pkg.small_memory)
    return pkg.switch_frame + 1
