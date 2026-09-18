"""Bounds and Layer 1: the baselines that transfer no latent memory.

Each baseline answers one question: what does the target model know at the
moment it takes over?

    bounds    warm_target, source_only, and first_only -- which is also Reset.
              SAM 2 does not know what to track until it is prompted, so a
              target that "starts with nothing" has to be given the original
              first-frame prompt; with nothing at all it scores ~0 and says
              nothing.  first_only is therefore the floor.
    layer 1   last_mask, first_plus_last, replay_k.  The source's tensors are
              discarded; only masks (as prompts) or re-watched frames travel.

`target_frames` -- post-switch frames the target processed and was scored on
-- is filled in by the runner, and is 0 for source_only.

Information budget: `last_mask`, `first_plus_last` and everything in Layer 2
may only use masks the *source model predicted*.  Ground truth enters exactly
once, as the first-frame prompt both models are given.  A baseline that quietly
reads a GT mask at the switch frame is not a baseline, it is an oracle, and it
would beat every method here for the wrong reason.

Layer 2 (latent transfer) lives in `baselines_latent.py` and registers into the
same table, so both layers are scored by one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .interfaces import MaskSet, Predictor, Translator

__all__ = ["BaselineResult", "Baseline", "BASELINES", "register", "add", "get",
           "replay_spec", "make_translated", "layer1_names", "REPLAY_KS",
           "LATENT_KINDS"]

#: SAM 2 keeps the prompted frame plus 6 recent mask memories, so from k=6 the
#: mask memory is full; object pointers reach back 15 frames (16 with the
#: prompted one), so k=16 is where replay approaches Warm Target.
REPLAY_KS = (1, 2, 4, 6, 8, 16)

#: Kinds that belong to Layer 2 and are excluded from `layer1_names`.
LATENT_KINDS = ("state", "oracle")


@dataclass
class BaselineResult:
    masks: dict[int, MaskSet]
    # False only for source_only: the source, not the target, produced the
    # post-switch frames, so the target processed none of them.
    by_target: bool = True
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Baseline:
    name: str
    kind: str
    description: str
    run: Callable[..., BaselineResult]


BASELINES: dict[str, Baseline] = {}


def register(name: str, kind: str, description: str, **kw):
    def deco(fn):
        BASELINES[name] = Baseline(name, kind, description, fn, **kw)
        return fn
    return deco


def add(baseline: Baseline) -> Baseline:
    """Register an already-built Baseline (Layer 2 builds them at run time,
    because a translator needs its fitted parameters before it exists)."""
    BASELINES[baseline.name] = baseline
    return baseline


def get(name: str) -> Baseline:
    if name not in BASELINES:
        raise KeyError("unknown baseline %r; have %s" % (name, sorted(BASELINES)))
    return BASELINES[name]


# ----------------------------------------------------------------- bounds
@register(
    "warm_target", "ceiling",
    "Upper bound. The large model runs the entire video itself from the "
    "original first-frame prompt; no switch happens. Retention is measured "
    "against this. It is a reference line rather than a hard ceiling -- a "
    "method can beat it on videos where the warm run drifted.",
)
def _warm_target(*, target_solo, switch, **_):
    # Independent of the switch frame, so the runner computes it once per video
    # and hands the same masks to all three switch points.  Recomputing it per
    # point tripled the runtime for byte-identical numbers.
    post = {f: m for f, m in target_solo.masks.items() if f > switch}
    return BaselineResult(post,
                          notes={"no_switch": True, "from_solo_run": True})


@register(
    "source_only", "ceiling",
    "The small model runs the entire video alone -- no switch, no target. "
    "Answers the prior question: is switching worth it? A handoff that scores "
    "below this one cost more than it bought.",
)
def _source_only(*, source_solo, switch, **_):
    post = {f: m for f, m in source_solo.masks.items() if f > switch}
    return BaselineResult(post, by_target=False,
                          notes={"no_switch": True, "source_finishes": True,
                                 "from_solo_run": True})


@register(
    "first_only", "floor",
    "Reset, and the lower bound. The large model is given only frame 0's image "
    "and its ground-truth mask -- the same start as the beginning of the video "
    "-- then jumps to switch+1. Everything the source accumulated is lost.",
)
def _first_only(*, target: Predictor, video, obj_ids, first_prompt, switch, **_):
    session = target.init_video(video, obj_ids)
    for oid, mask in first_prompt.items():
        target.add_prompt(session, 0, oid, mask=mask)
    preds = target.continue_from(session, switch + 1)
    return BaselineResult(preds)


# ---------------------------------------------------------------- layer 1
@register(
    "last_mask", "prompt",
    "The source's predicted mask on the last frame before the switch is given "
    "to the large model as a mask prompt on that frame. SAM 2 treats a prompt "
    "as ground truth, so a wrong source mask is locked in, and with no "
    "first-frame anchor an occlusion or a look-alike can pull it off target.",
)
def _last_mask(*, target: Predictor, video, obj_ids, source_masks, switch, **_):
    session = target.init_video(video, obj_ids)
    for oid, mask in source_masks[switch].items():
        target.add_prompt(session, switch, oid, mask=mask)
    preds = target.continue_from(session, switch + 1)
    return BaselineResult(preds)


@register(
    "first_plus_last", "prompt",
    "first_only and last_mask together: the frame-0 ground-truth prompt as the "
    "anchor, the source's mask on the last pre-switch frame as the current "
    "position.",
)
def _first_plus_last(*, target: Predictor, video, obj_ids, first_prompt,
                     source_masks, switch, **_):
    session = target.init_video(video, obj_ids)
    for oid, mask in first_prompt.items():
        target.add_prompt(session, 0, oid, mask=mask)
    for oid, mask in source_masks[switch].items():
        target.add_prompt(session, switch, oid, mask=mask)
    preds = target.continue_from(session, switch + 1)
    return BaselineResult(preds)


def _replay(k: int):
    def run(*, target: Predictor, video, obj_ids, first_prompt, switch, **_):
        session = target.init_video(video, obj_ids)
        for oid, mask in first_prompt.items():
            target.add_prompt(session, 0, oid, mask=mask)
        start = max(1, switch - k + 1)
        target.seek(session, start)
        replayed = target.run_until(session, switch)
        tail = target.continue_from(session, switch + 1)
        post = {f: m for f, m in tail.items() if f > switch}
        return BaselineResult(
            post,
            notes={"replay_k": k, "replay_from": start, "replayed": len(replayed)},
        )
    return run


def replay_spec(k: int) -> str:
    return "replay_%d" % k


for _k in REPLAY_KS:
    register(
        replay_spec(_k), "replay",
        "The large model is given the frame-0 ground-truth prompt, re-tracks "
        "the %d frame(s) immediately before the switch itself, then continues. "
        "From k=6 the mask memory is full; at "
        "k=16 the object-pointer window is too, and replay approaches Warm "
        "Target. At equal cost a method has to score higher, or match it for less." % _k,
    )(_replay(_k))


def make_translated(translator: Translator, name: str | None = None,
                    description: str | None = None) -> Baseline:
    """Every latent-transfer method -- Direct Copy, Norm-Matched, and any
    translator added later -- enters the study through this one function.

    Consequence worth stating out loud: there is no code path on which the
    proposed method is evaluated differently from the baseline it has to beat.
    Same session construction, same import, same continuation, same scoring.
    """

    def run(*, target: Predictor, video, obj_ids, source_state, switch, **_):
        session = target.init_video(video, obj_ids)
        translated = translator.translate(source_state)
        target.import_state(session, translated)
        preds = target.continue_from(session, switch + 1)
        return BaselineResult(
            preds,
            notes={"translator": translator.name,
                   "source_model": getattr(source_state, "model_id", "?"),
                   "slots": len(getattr(translated, "slots", []) or [])},
        )

    return Baseline(
        name=name or "translated::%s" % translator.name,
        kind="state",
        description=description or "%s -> %s via %s" % (
            translator.source_model, translator.target_model, translator.name),
        run=run,
    )


def layer1_names() -> list[str]:
    """Bounds and Layer 1 -- everything that does not transfer latent state."""
    return [n for n, b in BASELINES.items() if b.kind not in LATENT_KINDS]
