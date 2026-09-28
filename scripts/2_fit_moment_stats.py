"""[2] train fit 영상을 Small·Base+ 로 각각 돌려 기억 값의 평균·표준편차를 저장한다.

    python scripts/2_fit_moment_stats.py

결과: outputs/moment_stats.npz (Moment-Matched Copy 가 씀)
데이터셋마다 fit 영상을 최대 MOMENT_STATS_MAX_VIDEOS 개 골라, 영상마다 첫 객체 하나만 쓴다.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import settings  # noqa: E402
from benchmark import moment_stats  # noqa: E402
from benchmark.data import load_dataset, load_video_list  # noqa: E402
from benchmark.model import sam2_runner  # noqa: E402


def pick_items():
    items = []
    for dataset in settings.MOMENT_STATS_DATASETS:
        videos = {v.name: v for v in load_dataset(dataset)}
        entries = [e for e in load_video_list(dataset)["videos"]
                   if e["part"] == "fit" and e["objects"]]
        rng = random.Random(settings.MOMENT_STATS_SEED)
        chosen = rng.sample(entries, min(settings.MOMENT_STATS_MAX_VIDEOS, len(entries)))
        items += [(videos[e["video"]], e["objects"][0]) for e in chosen]
        print(f"{dataset}: fit 영상 {len(entries)}개 중 {len(chosen)}개 사용")
    return items


def main():
    items = pick_items()
    stats = {}
    for key in (settings.SOURCE_MODEL, settings.TARGET_MODEL):
        runner = sam2_runner.load_runner(key)
        stats[key] = moment_stats.collect(runner, items)
        del runner
    path = moment_stats.save(stats)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
