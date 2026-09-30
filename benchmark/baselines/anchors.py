"""기억 칸은 버리고, 마스크 몇 장만 Base+ 에 프롬프트로 주는 비교군.

Original-Prompt(s)     : 처음 받은 정답 마스크만. (1장 다시 봄)
Last-Visible           : Small이 마지막으로 "보인다"고 한 프레임의 Small 마스크만. (1장)
Original+Last-Visible  : 둘 다. (2장, 같은 프레임이면 1장)

Base+ 는 프롬프트 프레임 이미지를 한 번씩 다시 봐야 기억으로 만들 수 있다 → 그만큼 "다시 본 프레임".
"""


def original_prompt(session, pkg, stats):
    session.add_prompt(pkg.prompt_frame, pkg.prompt_mask)
    return pkg.switch_frame + 1, 1


def last_visible(session, pkg, stats):
    frame, mask = pkg.last_visible_or_prompt()
    session.add_prompt(frame, mask)
    return pkg.switch_frame + 1, 1


def original_last_visible(session, pkg, stats):
    session.add_prompt(pkg.prompt_frame, pkg.prompt_mask)
    frame, mask = pkg.last_visible_or_prompt()
    if frame == pkg.prompt_frame:
        return pkg.switch_frame + 1, 1
    session.add_prompt(frame, mask)
    return pkg.switch_frame + 1, 2
