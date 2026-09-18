# Metrics

Four columns, computed per (video, switch point, method, object) row and
averaged at analysis time.

| column | what it is |
|---|---|
| `jf_gtvis` | GT-visible J&F over every post-switch frame |
| `jf_at_5_gtvis` | Post-Switch J&F@5: the same, over the 5 frames right after the switch |
| `retention` | `mean(jf_gtvis) / mean(warm_jf_gtvis) x 100`, per method |
| `target_frames` | post-switch frames the large model processed and was scored on |

## GT-visible J&F

J is the overlap (IoU) between the predicted and ground-truth masks, F is the
accuracy of the boundary, and J&F is their mean.

J and F score a frame where both masks are empty as 1.0.  That is correct per
frame and poison in aggregate: on a video where the object is hidden for a
third of its length, a model that has lost the object and outputs nothing
collects a free 1.0 on every hidden frame.  So the numbers are computed only
over frames whose ground truth for *that object* is non-empty.

## Post-Switch J&F@5

GT-visible J&F over `[t_s+1, t_s+5]` -- the frames where a broken handoff shows
before the model recovers or drifts.  A window with no visible ground truth is
NaN and left out of the average, never scored 0.

## Retention

```
Retention = mean(jf_gtvis) / mean(warm_jf_gtvis) x 100
```

How much of Warm Target's score survived the switch.  A ratio of means over the
same rows, divided once at aggregation -- never a mean of per-row ratios, which
would let one video where the warm run scored 0.03 contribute a 2000%.  Values
above 100 are not clipped: SAM 2 is autoregressive, and a method that re-enters
with fresher information can beat a warm run that drifted.

## target_frames

The number of post-switch frames the large model processed and was scored on:
`num_frames - switch - 1` for every method, and 0 for `source_only`, where the
small model finishes the video instead.

## Information budget

`last_mask`, `first_plus_last` and every Layer 2 method use only masks the
*source model predicted*.  Ground truth enters exactly once, as the first-frame
prompt both models receive.
