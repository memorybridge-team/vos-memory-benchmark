"""Run the full harness on the synthetic dataset and assert the orderings that
must hold for any sane memory-based model.

This is a regression test for the *harness*, not evidence about SAM 2.  It runs
with no GPU, no checkpoints and no dataset download, so it is the thing to run
after touching metrics, baselines, adapters or the runner -- and the thing to
run first when a real run produces a number that looks wrong, to rule out the
scoring code before blaming the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from hoeval.baselines import BASELINES
from hoeval.baselines_latent import register_layer2
from hoeval.datasets import DavisLayout
from hoeval.fitting import collect_stats
from hoeval.protocol import build_manifest
from hoeval.results import save_records, summarize, to_frame
from hoeval.runner import evaluate_dataset
from tests.fake_predictor import FakePredictor
from tests.synthetic import make_dataset

COLS = ["method", "n", "jf_gtvis", "jf_at_5_gtvis", "retention", "target_frames"]


def main(out_dir: Path):
    root = out_dir / "synthetic"
    make_dataset(root)
    ds = DavisLayout(root, "synthetic")
    manifest = build_manifest("synthetic", ds.video_lengths())
    manifest.save(out_dir / "switch_manifest.json")

    src = FakePredictor("sam2_small", skill=0.85)
    tgt = FakePredictor("sam2_large", skill=1.00)

    # Same two calls the server makes, on a fixture instead of DAVIS train.
    stats = {m.model_id: collect_stats(m, ds, split="train", every=6)
             for m in (src, tgt)}
    layer2 = register_layer2(
        source_model=src.model_id, target_model=tgt.model_id,
        source_stats=stats[src.model_id], target_stats=stats[tgt.model_id],
    )

    records = evaluate_dataset(dataset=ds, manifest=manifest, source=src, target=tgt)
    save_records(records, out_dir / "records.csv")

    df = to_frame(records)
    summary = summarize(df).sort_values("jf_gtvis", ascending=False)
    print(summary[COLS].to_string(index=False, float_format=lambda v: "%.3f" % v))

    sm = summary.set_index("method")
    by = sm["jf_gtvis"]

    # Retention is a ratio of means; recomputing it the long way must agree.
    w = df[df["method"] == "warm_target"]
    retention_consistent = abs(
        sm.loc["first_only", "retention"]
        - 100.0 * df[df["method"] == "first_only"]["jf_gtvis"].mean()
        / w["jf_gtvis"].mean()
    ) < 1e-6

    post = df["num_frames"] - df["switch_frame"] - 1
    target_ran = df["method"] != "source_only"
    oracle = df[df["method"] == "re_encode_oracle"]
    reencoded_ok = all(
        n["reencoded"] == min(6, sw)
        for n, sw in zip(oracle["notes"], oracle["switch_frame"])
    )

    checks = [
        # --- scoring ------------------------------------------------------
        ("warm_jf_gtvis is the same for every method on a given row",
         df.groupby(["video", "fraction", "obj_id"])["warm_jf_gtvis"].nunique().eq(1).all()),
        ("warm_target's own retention is exactly 100",
         abs(sm.loc["warm_target", "retention"] - 100.0) < 1e-9),
        ("retention is a ratio of means, not a mean of ratios", retention_consistent),
        ("target_frames counts every post-switch frame the target produced",
         (df.loc[target_ran, "target_frames"] == post[target_ran]).all()),
        ("target_frames is 0 for source_only",
         (df.loc[~target_ran, "target_frames"] == 0).all()),
        # --- ordering -----------------------------------------------------
        ("replay is monotone in k",
         by["replay_1"] <= by["replay_4"] + 1e-9 <= by["replay_6"] + 1e-9),
        ("replay saturates at k=6 (SAM 2 keeps 6 recent memories)",
         abs(by["replay_6"] - by["replay_16"]) < 1e-9),
        ("source_only is the source's own quality, not the target's",
         abs(by["source_only"] - by["warm_target"]) > 1e-6),
        # --- layer 2 --------------------------------------------------------
        ("layer 2 registered direct_copy, norm_matched, re_encode_oracle",
         layer2 == ["direct_copy", "norm_matched", "re_encode_oracle"]),
        ("norm_matched actually reaches the tensors",
         abs(by["norm_matched"] - by["direct_copy"]) > 1e-6),
        ("re_encode_oracle re-encodes exactly the source's recent frames",
         reencoded_ok),
        ("re_encode_oracle is at least as good as a copy of the same memory",
         by["re_encode_oracle"] >= max(by["direct_copy"], by["norm_matched"]) - 1e-9),
        # --- protocol -------------------------------------------------------
        ("every method scored on identical switch frames",
         df.groupby(["video", "fraction"])["switch_frame"].nunique().eq(1).all()),
        ("one row per video x fraction x method x object",
         len(df) == df[["video", "fraction", "method", "obj_id"]].drop_duplicates().shape[0]),
        ("every row carries the manifest it was produced under",
         df["manifest_digest"].eq(manifest.digest).all()),
        ("all %d registered methods ran" % len(BASELINES),
         set(df["method"]) == set(BASELINES)),
    ]
    print("")
    ok = True
    for label, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + label)
        ok &= bool(passed)
    print("")
    print("%d records, %d methods -> %s"
          % (len(df), df["method"].nunique(), out_dir / "records.csv"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/selftest")))
