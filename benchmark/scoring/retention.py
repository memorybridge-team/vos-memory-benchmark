"""회복률 = 방법 ÷ Full Replay × 100.  영상마다 비율을 구한 뒤 평균.

순서:
  1. 영상마다 J&F 한 개로 모은다 (그 영상의 객체·전환 시점 줄들의 평균).
  2. 영상마다 방법 J&F ÷ Full Replay J&F × 100.
  3. 영상들의 비율을 평균.

기준 두 가지 (결과 줄의 열 이름):
  "jf_whole" 영상 전체 기준 — 전환 전은 Small, 전환 뒤는 방법의 결과. MOSEv2 서버 점수와 같은 기준.
  "jf_post"  전환 뒤 기준  — 전환 뒤, 정답에 객체가 보이는 프레임만. 방법 차이가 또렷함.

Full Replay 점수가 0인 영상은 나눌 수 없어 뺀다 (빠진 수는 dropped 로 알려줌).
"""

from __future__ import annotations

from collections import defaultdict


def video_means(rows: list[dict], key: str) -> dict[str, float]:
    """영상 → 그 영상 줄들의 key 평균 (값이 없는 줄은 건너뜀)."""
    values = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            values[r["video"]].append(r[key])
    return {video: sum(v) / len(v) for video, v in values.items()}


def retention_by_video(baseline_rows: list[dict], replay_rows: list[dict], key: str):
    """영상 → 회복률(%). 두 번째 값은 Full Replay 가 0이라 뺀 영상 수."""
    baseline = video_means(baseline_rows, key)
    replay = video_means(replay_rows, key)
    ratios, dropped = {}, 0
    for video, score in baseline.items():
        base = replay.get(video)
        if base is None:
            continue
        if base <= 0:
            dropped += 1
            continue
        ratios[video] = score / base * 100
    return ratios, dropped
