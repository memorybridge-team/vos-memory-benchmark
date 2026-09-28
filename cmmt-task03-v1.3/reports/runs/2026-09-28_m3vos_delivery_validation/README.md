# M³-VOS immutable delivery inventory

## Purpose

Verify that the completed Hugging Face delivery is usable as a CMMT external
benchmark before a prompt loader, switch manifest, or evaluator result is
claimed. This is an inventory gate, not a model evaluation.

## Source and reproducibility

- Dataset: `Lijiaxin0111/M3_VOS`
- Pinned revision: `5deb15b2baeaaa294ca168b789537729f7fb53a5`
- Delivery root: `/workspace/CMMT/data/M3VOS-manual`
- Validator: `scripts/validate_m3vos_inventory.py`
- Command:

  ```bash
  python3 scripts/validate_m3vos_inventory.py \
    /workspace/CMMT/data/M3VOS-manual \
    --revision 5deb15b2baeaaa294ca168b789537729f7fb53a5 \
    --output reports/tasks/03_benchmark/runs/2026-09-28_m3vos_delivery_validation/inventory.json
  ```

The direct file-level Hugging Face delivery has no single archive file to
checksum. The immutable revision plus the committed `inventory.json` digest is
therefore the reproducibility anchor for this received tree.

## Result

| Check | Result |
| --- | ---: |
| `data/ImageSets/val.txt` / JPEG / annotation / viewer metadata / target-object metadata set agreement | pass |
| sequences | 471 |
| RGB JPEG frames | 202,577 |
| GT PNG annotations | 202,577 |
| object records | 530 |
| `all_core_seqs.txt` members | 68 |
| RGB/annotation stem mismatches or metadata disagreement | 0 |

`inventory.json` SHA-256:
`2a6c98fede7e7d5ce6e85f4ae093a106842111a5906becc0abb7db6bbd73ad41`.

The validator originally treated `target_object.json` values as lists. The
received revision instead uses a mapping such as `{"obj_1": {...}}`; the
validator was corrected and the full scan rerun. This is a parser fix, not a
data redownload.

## Interpretation and remaining gate

The published paper/project materials quote 479 videos and 205,181 dense
masks, whereas this fixed Hugging Face delivery validates at 471 sequences and
202,577 RGB/GT pairs. CMMT must not silently substitute either number: reports
will cite the literature figure as context and use the received delivery's
inventory for manifests and denominator checks.

The official evaluator checkout (`M3VOS_Experiment` commit
`8cf8f9b3cb069d8476ef6c3c0b8f11b8337c3b56`) emits `J`, `J_last`, and `J_cc`.
`J_last` is the last quarter after its temporal sampling and endpoint removal;
it is not named `J_tr` in the released code. The evaluator identifies label
255 as void while its default invocation passes no void array to the metric
function. CMMT will preserve an **official-output** smoke and a separate
**void-aware engineering** check rather than conflate them.

Still required before Task 03 can close: object-level first-prompt loader and
fixed switch manifest, official evaluator GT-copy/shard-merge smoke, and the
conditional boundary F/J&F integrity gate.
