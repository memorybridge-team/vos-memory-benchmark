"""[추가] 진단용 비교군. 확정 표에는 넣지 않고 추가 표에만 나온다.

reset          : 아무것도 넘기지 않음. SAM2는 프롬프트가 없으면 무엇을 따라갈지 모르므로
                 전환 뒤 전부 빈 마스크 → "바닥" 확인용.
recent_k_only  : 처음 정답 없이 Small 마스크 한 장(s-K+1)에서 시작해 최근 K 프레임만 다시 봄.
                 Original+Replay-K 와 비교해 "처음 정답"이 얼마나 중요한지 본다.
"""

import settings


def reset(session, pkg, stats):
    return None


def recent_k_only(session, pkg, stats):
    s = pkg.switch_frame
    first = max(pkg.prompt_frame, s - settings.EXTRA_RECENT_K + 1)
    session.add_prompt(first, pkg.recent_masks[first])
    return first + 1
