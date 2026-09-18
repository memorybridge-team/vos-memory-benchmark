"""Freeze the switch points once, so every run in the study shares them.

    python scripts/build_manifest.py \
        --dataset-root /data/DAVIS --subdir 480p \
        --split /data/DAVIS/ImageSets/2017/val.txt \
        --out configs/davis_val_switches.json

Every result row carries this file's digest.  Rows produced under different
digests are not comparable and must never be averaged together -- which is
exactly the accident that happens when someone regenerates the manifest
halfway through a study and nothing complains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoeval.datasets import davis2017
from hoeval.protocol import DEFAULT_FRACTIONS, build_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis", required=True, help="unpacked DAVIS 2017 trainval root")
    ap.add_argument("--out", default="configs/davis_val_switches.json")
    a = ap.parse_args()

    ds = davis2017(a.davis, "val")
    lengths = ds.video_lengths()
    manifest = build_manifest(ds.name, lengths, DEFAULT_FRACTIONS)
    manifest.save(a.out)

    skipped = sorted(v for v, n in lengths.items() if n < 3)
    print("videos: %d   switch points: %d   digest: %s"
          % (len(lengths), len(manifest.points), manifest.digest))
    print("frames: min %d  mean %.1f  max %d"
          % (min(lengths.values()),
             sum(lengths.values()) / len(lengths),
             max(lengths.values())))
    if skipped:
        print("skipped (fewer than 3 frames): %s" % skipped)
    # A 75%% switch on a short video can leave fewer than 5 frames, which makes
    # Post-Switch J&F@5 undefined.  Report the count now rather than discovering
    # it as a pile of NaNs in the results table.
    tight = [p for p in manifest.points if p.num_frames - p.frame - 1 < 5]
    print("switch points with fewer than 5 post-switch frames: %d%s"
          % (len(tight), "" if not tight else "  -> @5 will be NaN for these"))
    print("wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
