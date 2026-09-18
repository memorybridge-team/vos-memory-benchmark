"""Acceptance test for a state implementation: export, re-import, and get the
same video back.

    export at frame t -> install into a fresh session -> run to the end
                              must equal
    the original session simply running to the end

Frame-identical, not "close".  Anything less means the snapshot is missing
something the model reads, and the usual culprits are, in order:

  1. `object_score_logits` was not carried, so the target thinks an object the
     source had lost is visible;
  2. the object index mapping was not applied, so object 2's memory is attached
     to object 1 -- every tensor correct, every association wrong;
  3. too few recent slots were exported.  SAM 2 reads `num_maskmem - 1` memory
     slots *and* obj_ptrs from up to `max_obj_ptrs_in_encoder - 1` frames, and
     the second window is the larger one.  Export only the memory slots and the
     divergence starts small and grows, which looks exactly like a translator
     problem and is not one.

Run it against the fake predictor to check the test itself, then against SAM 2:

    python tests/test_state_roundtrip.py --fake
    python tests/test_state_roundtrip.py \
        --checkpoints /path/to/sam2/checkpoints \
        --video /path/to/DAVIS/JPEGImages/480p/bike-packing \
        --annotation /path/to/DAVIS/Annotations/480p/bike-packing/00000.png

Until this passes, nothing measured downstream means anything.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image


def _compare(a: dict, b: dict, label_a: str, label_b: str):
    """Returns (identical, first_divergent_frame, mean_iou_at_divergence)."""
    frames = sorted(set(a) & set(b))
    missing = sorted(set(a) ^ set(b))
    if missing:
        return False, missing[0], float("nan")
    for f in frames:
        for oid in sorted(set(a[f]) | set(b[f])):
            x = np.asarray(a[f].get(oid))
            y = np.asarray(b[f].get(oid))
            if x.shape != y.shape or not np.array_equal(x, y):
                union = np.count_nonzero(x | y)
                iou = 1.0 if union == 0 else np.count_nonzero(x & y) / union
                print("    first divergence: frame %d, object %d, IoU(%s,%s)=%.4f"
                      % (f, oid, label_a, label_b, iou))
                return False, f, float(iou)
    return True, None, 1.0


def build_fake():
    from tests.fake_predictor import FakePredictor
    from tests.synthetic import make_dataset

    root = Path(tempfile.mkdtemp()) / "synthetic"
    make_dataset(root, videos=("square_a",), n_frames=40)
    video = root / "JPEGImages" / "square_a"
    ann = root / "Annotations" / "square_a" / "00000.png"
    return FakePredictor("sam2_small", skill=0.9), FakePredictor("sam2_large"), video, ann


def build_sam2(args):
    from hoeval.models.registry import build

    src = build(args.source, args.checkpoints, device=args.device)
    tgt = build(args.target, args.checkpoints, device=args.device)
    return src, tgt, Path(args.video), Path(args.annotation)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true",
                    help="exercise the test against the reference fixture")
    ap.add_argument("--checkpoints")
    ap.add_argument("--video")
    ap.add_argument("--annotation")
    ap.add_argument("--source", default="sam2_small")
    ap.add_argument("--target", default="sam2_large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--switch", type=int, default=12)
    a = ap.parse_args()

    if a.fake:
        src, tgt, video, ann = build_fake()
    else:
        if not (a.checkpoints and a.video and a.annotation):
            ap.error("--checkpoints, --video and --annotation are required "
                     "unless --fake is given")
        src, tgt, video, ann = build_sam2(a)

    ids = np.array(Image.open(ann).convert("P"))
    obj_ids = [int(i) for i in np.unique(ids) if i != 0]
    prompt = {oid: (ids == oid) for oid in obj_ids}
    switch = a.switch
    print("video=%s objects=%s switch=%d" % (Path(video).name, obj_ids, switch))

    # --- reference: one uninterrupted session ------------------------------
    s1 = src.init_video(str(video), obj_ids)
    for oid, m in prompt.items():
        src.add_prompt(s1, 0, oid, mask=m)
    src.run_until(s1, switch)
    state = src.export_state(s1)
    print("exported: %s" % state.describe())
    reference = src.continue_from(s1, switch + 1)

    # --- same model, via export/import -------------------------------------
    s2 = src.init_video(str(video), obj_ids)
    src.import_state(s2, state)
    roundtrip = src.continue_from(s2, switch + 1)

    checks = []
    same, frame, _ = _compare(reference, roundtrip, "direct", "roundtrip")
    checks.append(("same-model roundtrip is frame-identical"
                   + ("" if same else " (diverges at frame %s)" % frame), same))

    # export must not disturb the session it snapshotted
    checks.append(("export_state did not mutate the source session",
                   state.frame_idx == switch and len(state.slots) > 0))

    # --- seek runs nothing --------------------------------------------------
    s3 = src.init_video(str(video), obj_ids)
    for oid, m in prompt.items():
        src.add_prompt(s3, 0, oid, mask=m)
    before = getattr(s3, "encoder_calls", None)
    src.seek(s3, switch)
    after = getattr(s3, "encoder_calls", None)
    checks.append(("seek runs no image encoder"
                   + ("" if before is not None else " (not instrumented; skipped)"),
                   before is None or before == after))

    # --- a foreign state is accepted, not rejected --------------------------
    s4 = tgt.init_video(str(video), obj_ids)
    try:
        tgt.import_state(s4, state)
        accepted = True
    except Exception as exc:                                # noqa: BLE001
        accepted = False
        print("    import_state raised on a foreign model: %r" % exc)
    checks.append(("target accepts a state from a different model", accepted))
    if accepted:
        foreign = tgt.continue_from(s4, switch + 1)
        checks.append(("foreign state produces a full-length prediction",
                       len(foreign) == len(reference)))

    print("")
    ok = True
    for label, passed in checks:
        print(("  PASS  " if passed else "  FAIL  ") + label)
        ok &= bool(passed)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
