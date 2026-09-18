"""Fit `norm_matched`: each model's own per-channel statistics, on train.

    python scripts/fit_norm_stats.py \
        --dataset-root /data/DAVIS --subdir 480p \
        --split /data/DAVIS/ImageSets/2017/train.txt \
        --checkpoints /path/to/sam2/checkpoints \
        --models sam2_small,sam2_large \
        --out configs/stats

No pairing happens here, and that is the point.  Each model runs the split on
its own and reports what its memory looks like; `norm_matched` then puts two
independent summaries side by side.  There is no correspondence to build.

**Train only**, and not configurable: val videos are the evaluation, and a
channel mean is still information taken out of them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoeval.datasets import davis2017
from hoeval.fitting import collect_stats
from hoeval.models.registry import build


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", required=True, help="unpacked DAVIS 2017 trainval root")
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--models", default="sam2_small,sam2_large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--every", type=int, default=8,
                    help="export the state every N frames; the recent bank holds "
                         "only a handful of slots, so one export per video would "
                         "describe its ending and nothing else")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="configs/stats")
    a = ap.parse_args()

    ds = davis2017(a.davis, "train")
    out = Path(a.out)
    for name in [m.strip() for m in a.models.split(",") if m.strip()]:
        model = build(name, a.checkpoints, device=a.device)
        seen = {"n": 0}

        def progress(video, added, seen=seen):
            seen["n"] += added
            print("  %-28s +%-5d slots  (%d total)" % (video, added, seen["n"]),
                  flush=True)

        print("fitting %s on %d videos" % (name, len(ds.video_names)))
        book = collect_stats(model, ds, split="train", every=a.every,
                             limit=a.limit, on_video=progress)
        book.meta["checkpoint"] = model.checkpoint
        book.meta["fingerprint"] = model.fingerprint
        path = out / ("%s.json" % name)
        book.save(path)
        for group, frozen in book.groups.items():
            print("  %-10s channels=%-4d values/channel=%-9d  mean|%.3f|  std %.3f"
                  % (group, len(frozen.mean), frozen.count,
                     abs(frozen.mean).mean(), frozen.std.mean()))
        print("wrote %s\n" % path)
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
