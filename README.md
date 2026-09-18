# hoeval — handoff evaluation harness (track B)

SAM2-S tracks a DAVIS 2017 video, hands over to SAM2-L at 25 / 50 / 75% of its
length, and every baseline below is scored on the same switch points with the
same four metrics.

```bash
python tests/test_end_to_end.py runs/selftest    # no GPU, no checkpoints, no dataset
python tests/test_state_roundtrip.py --fake      # the acceptance test, against the fixture
```

## Run order on the server

```bash
# 1. what does the installed SAM 2 actually store?   -> docs/sam2_state_layout.md
python scripts/dump_sam2_state.py --checkpoints $CKPT \
    --video $DAVIS/JPEGImages/480p/bike-packing \
    --annotation $DAVIS/Annotations/480p/bike-packing/00000.png

# 2. is export/import lossless?   frame-identical or it is not done
python tests/test_state_roundtrip.py --checkpoints $CKPT \
    --video $DAVIS/JPEGImages/480p/bike-packing \
    --annotation $DAVIS/Annotations/480p/bike-packing/00000.png

# 3. freeze the switch points, once, for the whole study
python scripts/build_manifest.py --davis $DAVIS

# 4. norm_matched's statistics -- train split only
python scripts/fit_norm_stats.py --davis $DAVIS --checkpoints $CKPT

# 5. run it first on 5 videos: if a verbatim copy already lands near Warm
#    Target, there is no representation gap inside SAM 2
python scripts/run_davis.py --davis $DAVIS --checkpoints $CKPT \
    --methods direct_copy,warm_target --limit 5 --out runs/probe

# 6. the run
python scripts/run_davis.py --davis $DAVIS --checkpoints $CKPT \
    --stats configs/stats --out runs/davis_s2l
python scripts/analyze.py --records runs/davis_s2l/records.csv --out runs/davis_s2l
```

## SAM 2 memory

- mask memory: the prompted frame plus the 6 most recent frames
- object pointers: up to 16 (the prompted frame plus 15 recent)

## Methods

### Bounds

| name | what happens |
|---|---|
| `warm_target` | Upper bound. No switch: the large model runs the whole video from frame 0. |
| `source_only` | The small model runs the whole video alone. If switching scores below this, it cost more than it bought. |
| `first_only` | **Reset**, the lower bound. SAM 2 needs a prompt to know what to track, so "starting with nothing" means the large model gets only frame 0's image and ground-truth mask. |

### Layer 1 — no memory transfer

The small model's memory tensors are discarded; only masks (as prompts) or
re-watched frames travel.

| name | what the large model gets |
|---|---|
| `first_only` | frame 0 image + ground-truth mask (see Bounds) |
| `last_mask` | the last pre-switch frame + the small model's predicted mask on it |
| `first_plus_last` | both of the above |
| `replay_{1,2,4,6,8,16}` | the frame-0 prompt, then re-tracks the last k pre-switch frames itself |

SAM 2 treats a prompt as ground truth, so `last_mask` locks in any error in the
small model's mask, and without a frame-0 anchor it can drift to a look-alike.
From k=6 replay fills the mask memory; at k=16 it fills the object-pointer
window too and approaches Warm Target.

### Layer 2 — memory transfer

| name | what the large model gets |
|---|---|
| `direct_copy` | the small model's memory tensors, verbatim — possible because SAM 2's memory is 64 channels at every size |
| `norm_matched` | the same, after per-channel mean/std matching (statistics from the train split) |
| `re_encode_oracle` | the frames the small model remembered, re-encoded by the large model's own encoders with the small model's masks |

`re_encode_oracle` is a reference point, not a method: it reads past frames.
Unlike `replay_k` it does not re-predict anything — it is what a perfect
translation of the small model's memory into the large model's representation
would score.  It carries exactly the frames `direct_copy` hands over, so the gap
between the two is the representation gap.

## Metrics

| column | one line |
|---|---|
| `jf_gtvis` | GT-visible J&F: J&F only over post-switch frames where the object is visible in the ground truth |
| `jf_at_5_gtvis` | Post-Switch J&F@5: the same over the 5 frames after the switch; NaN when nothing is visible, never 0 |
| `retention` | post-switch score as a % of Warm Target's: `mean(jf_gtvis) / mean(warm_jf_gtvis) x 100` |
| `target_frames` | post-switch frames the large model processed and was scored on (0 for `source_only`) |

Details: `docs/METRICS.md`.

## What track A must implement

`hoeval.interfaces.Predictor`.  Nothing outside `hoeval/models/` imports SAM 2.

| method | contract |
|---|---|
| `init_video(video_dir, obj_ids)` | fresh session, empty memory |
| `add_prompt(session, frame_idx, obj_id, mask=/points=/box=)` | prompt one frame |
| `seek(session, frame_idx)` | move the read position **without processing a frame and without touching memory** |
| `run_until(session, frame_idx)` | propagate through `frame_idx`, accumulating memory |
| `continue_from(session, frame_idx)` | propagate to the end, no re-processing before `frame_idx` |
| `encode_memory(session, frame_idx, masks)` | write one frame into memory from given masks with this model's encoders, predicting nothing |
| `export_state(session)` | snapshot the accumulated memory |
| `import_state(session, state)` | install a snapshot, **including one whose `model_id` differs** |

`hoeval/models/sam2_wrapper.py` is the SAM 2 implementation;
`tests/fake_predictor.py` is a reference implementation with no weights.

A SAM 2 state carries `maskmem_features`, `obj_ptr`, `object_score_logits` (the
occlusion state) and the object index mapping.  Position encodings do not
travel: the large model re-issues its own.

## Switch-point protocol

`build_manifest` resolves one switch frame per (video, fraction) at 25/50/75%
and hashes the result.  Every record carries `manifest_digest`, and
`scripts/analyze.py` refuses to aggregate rows from different manifests.

## Results

One row per (video, fraction, method, object).  Aggregation happens at analysis
time.
