"""Fitting translator parameters from model runs, without keeping the tensors.

This serves `norm_matched`, which needs nothing but each model's own
per-channel statistics.  They are accumulated in a streaming pass: one SAM 2
memory is 262,144 floats and a split's worth of them does not fit anywhere
useful, while the per-channel summaries they reduce to do.

Split discipline
----------------
Every fit runs on **train**.  The val videos are the evaluation, and a statistic
as small as a channel mean is still information leaking out of them.  `StatsBook`
records the split it was fitted on and `NormMatched` refuses anything that does
not say "train", so the rule is enforced rather than remembered.
"""

from __future__ import annotations

from pathlib import Path

from .adapters.stats import ChannelStats, GROUPS, StatsBook, group_of
from .datasets import DavisLayout, GtCache
from .models.state import HandoffState
from .runner import solo_run

__all__ = ["accumulate_state", "collect_stats"]


def accumulate_state(acc: dict[str, ChannelStats], state: HandoffState, seen: set) -> int:
    """Fold one exported state into the accumulators; returns slots added.

    `seen` de-duplicates across checkpoints: a memory slot that survives from
    one export to the next is the same tensor, and counting it twice would
    quietly weight the middle of every video more heavily than its ends.
    """
    added = 0
    for slot in state.slots:
        key = (state.model_id, slot.obj_id, slot.frame_idx, slot.kind)
        if key in seen:
            continue
        seen.add(key)
        acc[group_of(slot.kind, "maskmem")].update(slot.maskmem)
        acc[group_of(slot.kind, "obj_ptr")].update(slot.obj_ptr)
        added += 1
    return added


def collect_stats(
    predictor,
    dataset: DavisLayout,
    *,
    split: str = "train",
    every: int = 8,
    limit: int | None = None,
    on_video=None,
) -> StatsBook:
    """Run `predictor` alone over the split and summarise the memory it builds.

    Memory is sampled by exporting the state every `every` frames rather than
    at the end: SAM 2's recent bank holds only 6 slots, so a single export at
    the last frame would describe the end of each video and nothing else.
    """
    acc = {g: ChannelStats() for g in GROUPS}
    seen: set = set()
    names = dataset.video_names[:limit] if limit else dataset.video_names
    for name in names:
        spec = dataset.video(name)
        gt = GtCache(spec)
        checkpoints = tuple(range(every, spec.num_frames, every)) or (spec.num_frames - 1,)
        run = solo_run(predictor, spec, gt.masks(0), checkpoints)
        added = sum(accumulate_state(acc, st, seen) for st in run.states.values())
        if on_video:
            on_video(name, added)
    return StatsBook.from_accumulators(
        predictor.model_id, split, acc,
        dataset=dataset.name, videos=len(names), every=every,
    )
