"""Original-Prompt(s)+Replay-8: 처음 정답 마스크 + 전환 직전 K 프레임을 Base+ 가 직접 다시 본다.

Base+ 는 처음 프롬프트를 기억으로 들고, 프레임 s-K+1 ~ s 를 스스로 추적해 자기 기억을 채운 뒤
s+1 부터 이어간다. 다시 보는 동안 낸 마스크는 버린다 (그 구간 결과는 이미 Small이 냈음).
유효한 객체만 입력받는다. 짧은 구간은 축소 실행하지 않고 거절한다.
"""


import settings


def make_replay(k: int):
    def original_replay(session, pkg):
        pre_frames = pkg.switch_frame - pkg.prompt_frame
        if pre_frames <= settings.MIN_TRACK_FRAMES or pre_frames < k:
            raise ValueError('Replay 준비 전 구간이 부족합니다. 객체의 모든 전환/방법을 제외하세요.')
        session.add_prompt(pkg.prompt_frame, pkg.prompt_mask)
        return pkg.switch_frame - k + 1
    original_replay.__name__ = f"original_replay_{k}"
    return original_replay
