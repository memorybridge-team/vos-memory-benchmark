"""Original + Last-Visible: 처음 정답과 Small의 마지막 비어 있지 않은 예측을 프롬프트로 준다.

각 마스크는 해당 프레임 이미지에서 Base+ 기억으로 인코딩한다. Small 기억은 넘기지 않는다.
마지막 예측이 처음 프레임이거나 없으면 처음 프롬프트 한 장만 넣는다.
"""


def original_last_visible(session, pkg):
    session.add_prompt(pkg.prompt_frame, pkg.prompt_mask)
    frame, mask = pkg.last_visible_or_prompt()
    if frame != pkg.prompt_frame:
        session.add_prompt(frame, mask)
    return pkg.switch_frame + 1
