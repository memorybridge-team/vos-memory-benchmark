"""기억 칸은 버리고, 마스크 몇 장만 Base+ 에 프롬프트로 주는 비교군.

Original-Prompt(s)     : 처음 받은 정답 마스크만.
Last-Visible           : Small이 마지막으로 "보인다"고 한 프레임의 Small 마스크만.
Original+Last-Visible  : 둘 다.
"""
# session: Base+가 받을 것
# pkg: 전환 할 때 Small에서 가져올 것
# stats: Moment-Matched Copy에 쓸 평균, 표준편차

# Original Prompt : memory bank없이 mask만 프롬프트로
def original_prompt(session, pkg, stats):
    session.add_prompt(pkg.prompt_frame, pkg.prompt_mask)
    return pkg.switch_frame + 1

# Last Visible :
def last_visible(session, pkg, stats):
    frame, mask = pkg.last_visible_or_prompt()
    session.add_prompt(frame, mask)
    return pkg.switch_frame + 1


def original_last_visible(session, pkg, stats):
    session.add_prompt(pkg.prompt_frame, pkg.prompt_mask)
    frame, mask = pkg.last_visible_or_prompt()
    if frame != pkg.prompt_frame:
        session.add_prompt(frame, mask)
    return pkg.switch_frame + 1
