"""Step 2-1: print what the installed SAM 2 actually stores, and check the
assumptions the wrapper is built on.

Run this FIRST, before the roundtrip test and long before DAVIS.  Everything
downstream -- the export/import, the translators, the paired-state pipeline --
rests on a handful of facts about `inference_state` that are cheap to verify now
and expensive to discover after a 3,000-run job produced numbers that mean
nothing.

    python scripts/dump_sam2_state.py \
        --checkpoints /path/to/sam2/checkpoints \
        --video /path/to/DAVIS/JPEGImages/480p/bike-packing \
        --annotation /path/to/DAVIS/Annotations/480p/bike-packing/00000.png \
        --models sam2_small,sam2_large \
        --out docs/sam2_state_layout.md

What it checks, and why each one matters if it fails:

  layout             the keys exist at all; otherwise the wrapper is guessing.
  occlusion state    object_score_logits is present per frame.  Absent, the
                     target reads a lost object as visible.
  index mapping      obj_id_to_idx exists.  Absent, transferred memory can land
                     on the wrong object with no error.
  pos-enc is shared  maskmem_pos_enc is one cached tensor, not per-frame data.
                     If it turns out to be per-frame, `import_state` must stop
                     regenerating it and the plan's step 2-4 needs revisiting.
  S/L token match    both models use the same memory grid and channel count.
                     This is the precondition for direct_copy and
                     norm_matched: the small model's tensors must fit the
                     large model's memory as they are.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from hoeval.models.registry import MODELS, build
from hoeval.models.sam2_layout import probe_layout


def _t(x):
    if hasattr(x, "shape"):
        return {"shape": list(x.shape), "dtype": str(x.dtype),
                "device": str(getattr(x, "device", "-"))}
    if isinstance(x, (list, tuple)):
        return {"list_of": len(x), "first": _t(x[0]) if x else None}
    return {"type": type(x).__name__, "value": repr(x)[:60]}


def inspect(name, ckpt_dir, video, ann_path, device, frames):
    pred = build(name, ckpt_dir, device=device)
    ids = np.array(Image.open(ann_path).convert("P"))
    obj_ids = [int(i) for i in np.unique(ids) if i != 0]

    session = pred.init_video(video, obj_ids)
    for oid in obj_ids:
        pred.add_prompt(session, 0, oid, mask=(ids == oid))
    pred.run_until(session, frames)

    st = session.inference_state
    layout = probe_layout(st)
    per_obj = st["output_dict_per_obj"]
    obj_idx = sorted(per_obj)[0]
    cond = per_obj[obj_idx]["cond_frame_outputs"]
    noncond = per_obj[obj_idx]["non_cond_frame_outputs"]
    sample = cond[sorted(cond)[0]]

    # Is maskmem_pos_enc one shared tensor, or per-frame content?
    pos_encs = [v.get("maskmem_pos_enc") for v in noncond.values()]
    pos_encs = [p[0] if isinstance(p, (list, tuple)) and p else p for p in pos_encs]
    pos_encs = [p for p in pos_encs if p is not None]
    shared = None
    if len(pos_encs) > 1:
        shared = bool(all(pos_encs[0].shape == p.shape and
                          bool((pos_encs[0] == p).all()) for p in pos_encs[1:]))

    state = pred.export_state(session)
    mm = next((s.maskmem for s in state.slots if s.maskmem is not None), None)
    ptr = state.slots[0].obj_ptr if state.slots else None

    info = {
        "model_id": name,
        "checkpoint": pred.checkpoint,
        "fingerprint": pred.fingerprint,
        "reads": pred.reads,
        "recent_slots_exported": pred.recent_slots,
        "inference_state_keys": list(layout.keys),
        "frame_out_keys": list(layout.frame_out_keys),
        "layout": {
            "output_dict_per_obj": layout.per_obj,
            "combined_output_dict": layout.combined,
            "frames_tracked_per_obj": layout.frames_tracked_per_obj,
            "obj_id_to_idx": layout.obj_maps,
        },
        "notes": layout.notes,
        "frame_output_sample": {k: _t(v) for k, v in sample.items()},
        "counts": {
            "objects": len(per_obj),
            "cond_frames": len(cond),
            "non_cond_frames": len(noncond),
            "slots_exported": len(state.slots),
        },
        "maskmem_shape": None if mm is None else list(np.asarray(mm.shape)),
        "obj_ptr_shape": None if ptr is None else list(np.asarray(ptr.shape)),
        "maskmem_pos_enc_identical_across_frames": shared,
        "encoder_calls": session.encoder_calls,
    }
    del pred
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--video", required=True, help="a JPEGImages/<video> directory")
    ap.add_argument("--annotation", required=True, help="that video's 00000.png")
    ap.add_argument("--models", default="sam2_small,sam2_large")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--out", default="docs/sam2_state_layout.md")
    a = ap.parse_args()

    names = [m.strip() for m in a.models.split(",") if m.strip()]
    for n in names:
        if n not in MODELS:
            raise SystemExit("unknown model %r; have %s" % (n, sorted(MODELS)))

    infos = [inspect(n, a.checkpoints, a.video, a.annotation, a.device, a.frames)
             for n in names]

    checks = []
    for i in infos:
        checks.append(("%s: per-object output dict" % i["model_id"],
                       i["layout"]["output_dict_per_obj"]))
        checks.append(("%s: object index mapping present" % i["model_id"],
                       i["layout"]["obj_id_to_idx"]))
        checks.append(("%s: object_score_logits stored per frame" % i["model_id"],
                       "object_score_logits" in i["frame_out_keys"]))
        checks.append(("%s: maskmem_pos_enc is one shared tensor "
                       "(import_state may regenerate it)" % i["model_id"],
                       i["maskmem_pos_enc_identical_across_frames"] is not False))
    if len(infos) == 2:
        checks.append((
            "%s and %s share a memory grid and channel count "
            "(precondition for direct_copy / norm_matched)"
            % (infos[0]["model_id"], infos[1]["model_id"]),
            infos[0]["maskmem_shape"] == infos[1]["maskmem_shape"]
            and infos[0]["obj_ptr_shape"] == infos[1]["obj_ptr_shape"],
        ))

    lines = ["# SAM 2 `inference_state` layout", "",
             "Generated by `scripts/dump_sam2_state.py` against the installed "
             "build. Everything in `hoeval/models/sam2_wrapper.py` is written "
             "against exactly this; regenerate after any sam2 upgrade.", ""]
    lines += ["## Assumption checks", ""]
    for label, ok in checks:
        lines.append("- %s **%s**" % ("PASS" if ok else "FAIL", label))
    lines += ["", "## Per model", ""]
    for i in infos:
        lines += ["### %s" % i["model_id"], "", "```json",
                  json.dumps(i, indent=2, default=str), "```", ""]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join("  %s  %s" % ("PASS" if ok else "FAIL", l) for l, ok in checks))
    print("\nwrote %s" % out)
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
