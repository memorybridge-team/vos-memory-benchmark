"""train 영상을 fit(통계 계산용)과 dev(개발 중 확인용)로 나눈다.

영상 이름만 보고 정하므로 몇 번을 돌려도, 어느 컴퓨터에서 돌려도 같은 결과가 나온다.
"""

import hashlib

import settings


def part_of(video_name: str) -> str:
    h = int(hashlib.md5(video_name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < settings.DEV_FRACTION else "fit"
