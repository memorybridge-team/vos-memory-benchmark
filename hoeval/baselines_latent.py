"""Layer 2: methods that hand the target the source's *latent* memory.

    direct_copy        the small model's memory tensors installed verbatim.
    norm_matched       the same, after per-channel mean/std matching.
    re_encode_oracle   the reference point for this layer, not a method: the
                       large model re-encodes the frames the source remembered,
                       with the masks the source predicted, using its own
                       encoders.  It is what a perfect translation of the
                       source's memory into the target's representation would
                       score.

Read `re_encode_oracle` against `direct_copy`: both carry the same frames and
the same source information, and differ only in whose encoder produced the
memory -- so the gap between them is the representation gap, the thing a
translator has to close.  Read it against `replay_k` too: replay has the target
*re-predict* those frames, the oracle keeps the source's masks.
"""

from __future__ import annotations

from .adapters import DirectCopy, NormMatched, StatsBook
from .baselines import BaselineResult, add, make_translated, register
from .interfaces import Predictor
from .models.state import SLOT_RECENT

__all__ = ["LAYER2_NOFIT", "register_layer2", "layer2_names"]

#: Registered unconditionally, except `norm_matched`, which only appears once
#: statistics are supplied: a silently-unfitted translator that quietly behaves
#: like Direct Copy would be the worst possible bug to have in a results table.
LAYER2_NOFIT = ("direct_copy", "norm_matched", "re_encode_oracle")


@register(
    "re_encode_oracle", "oracle",
    "Reference point for Layer 2, not a usable method: it reads past frames. "
    "The large model is given the frame-0 prompt, then re-encodes every recent "
    "frame the source's memory held, with the mask the source predicted on it, "
    "through its own image and memory encoders -- nothing is re-predicted. "
    "The score of a perfect translation of the source's memory.",
)
def _re_encode_oracle(*, target: Predictor, video, obj_ids, first_prompt,
                      source_masks, source_state, switch, **_):
    session = target.init_video(video, obj_ids)
    for oid, mask in first_prompt.items():
        target.add_prompt(session, 0, oid, mask=mask)
    # Exactly the frames direct_copy hands over, so the two differ only in
    # whose encoder wrote the memory.
    frames = sorted({s.frame_idx for s in source_state.of_kind(SLOT_RECENT)})
    for f in frames:
        target.encode_memory(session, f, source_masks[f])
    preds = target.continue_from(session, switch + 1)
    return BaselineResult(
        preds,
        notes={"reencoded": len(frames),
               "reencoded_from": frames[0] if frames else None},
    )


def register_layer2(
    *,
    source_model: str = "*",
    target_model: str = "*",
    source_stats: StatsBook | str | None = None,
    target_stats: StatsBook | str | None = None,
) -> list[str]:
    """Build and register Layer 2. Returns the names actually registered.

    `norm_matched` is skipped -- not stubbed, skipped -- when statistics are
    missing, so a run without them produces a results table that is visibly
    short a row rather than one with a row that means something else.
    """
    names: list[str] = []

    add(make_translated(
        DirectCopy(target_model=target_model, source_model=source_model),
        name="direct_copy",
        description="The small model's memory tensors installed in the large "
                    "model verbatim. Zero parameters. If this works, no "
                    "translator is needed and the study moves to a "
                    "heterogeneous model pair.",
    ))
    names.append("direct_copy")

    if source_stats is not None and target_stats is not None:
        src = source_stats if isinstance(source_stats, StatsBook) else StatsBook.load(source_stats)
        tgt = target_stats if isinstance(target_stats, StatsBook) else StatsBook.load(target_stats)
        add(make_translated(
            NormMatched(src, tgt), name="norm_matched",
            description="Per-channel mean/std matching before the copy, each "
                        "side measured independently on the train split. The "
                        "null hypothesis for every fitted translator.",
        ))
        names.append("norm_matched")

    names.append("re_encode_oracle")
    return names


def layer2_names(registered: list[str] | None = None) -> list[str]:
    return list(registered) if registered is not None else list(LAYER2_NOFIT)
