# SAM 2 `inference_state` layout

**Not yet generated.**  This file is produced by running

```bash
python scripts/dump_sam2_state.py \
    --checkpoints /path/to/sam2/checkpoints \
    --video      /data/DAVIS/JPEGImages/480p/bike-packing \
    --annotation /data/DAVIS/Annotations/480p/bike-packing/00000.png \
    --models sam2_small,sam2_large \
    --out docs/sam2_state_layout.md
```

on the machine that has SAM 2 installed.  Run it **before** the roundtrip test
and long before DAVIS, and commit whatever it writes here.

## Why it is the first step

`hoeval/models/sam2_wrapper.py` is written against a specific shape of
`inference_state`.  That shape differs between SAM 2 and SAM 2.1, and a wrapper
built on the wrong one does not crash — it exports a state that is missing
entries and produces numbers that look plausible.  Every version-dependent
assumption is therefore stated in one place, `hoeval/models/sam2_layout.py`, and
checked against the live object rather than remembered.

## What the dump checks, and what a failure means

| check | if it fails |
|---|---|
| `output_dict_per_obj` exists | the wrapper cannot export per-object memory; adapt `sam2_layout.py` before anything else |
| `obj_id_to_idx` / `obj_idx_to_id` exist | memory can be attached to the wrong object — every tensor numerically perfect, every association wrong, and nothing raises |
| `object_score_logits` stored per frame | the occlusion state is not travelling; the target reads an object the source had already lost as plainly visible |
| `maskmem_pos_enc` identical across frames | it is a cached constant and `import_state` may regenerate it from the target. If this comes back **False**, it is per-frame content and step 2-4 of the plan needs rethinking before any translator is fitted |
| S and L share memory grid and channel count | the small model's tensors do not fit the large model's memory as they are, and `direct_copy` / `norm_matched` cannot run |

The dump also records, per model:

- `num_maskmem` and `max_obj_ptrs_in_encoder` — how far back the model actually
  reads, which is what `export_state` ships.  Note these are usually **different
  windows** (6 memory slots, 15 obj_ptr frames) and the export has to cover the
  larger one or the roundtrip test diverges in a way that looks like a
  translator problem.
- the checkpoint fingerprint, so a results table can be traced to the weights
  that produced it.  SAM 2 and SAM 2.1 ship checkpoints with interchangeable
  names and different memory behaviour; a mixed pair is wrong in a way no metric
  will flag.
