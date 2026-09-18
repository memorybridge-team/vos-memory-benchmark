"""Harness: every method sees the same video, the same switch frame, and the
same source run.

Two invariants this file exists to hold:

1. **One source run per (video, switch point), shared by every method.**  If
   each method re-ran the source, nondeterminism would leak straight into the
   comparison it is track B's job to keep clean.

2. **No method is scored on a different frame set than another.**  Curves come
   from the same GT cache, over the same post-switch window, per object.

Where the time goes
-------------------
The naive shape of this loop -- for each switch point, run the source to the
switch, then run the target warm over the whole video -- spends most of its
budget recomputing identical things.  Warm Target and Source Only do not depend
on the switch frame at all, and the source's run to a 25% switch is a strict
prefix of its run to the 75% one.

So the unit of work here is the **solo run**: one model, one video, prompted on
frame 0, propagated to the end, exporting its state as it passes each switch
frame.  Propagation is deterministic and autoregressive, so a state exported
mid-pass is identical to one from a run that stopped there -- the saving is
exact, not an approximation.

One solo run per (model, video) then serves, for a 3-switch-point manifest:

    the target's   warm_target    x3   (slice by switch)
    the source's   source_only    x3
    the source's   source_masks   x3   (prefix)
    the source's   source_state   x3   (exported in-pass)

That is 2 solo model-passes per video instead of 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .baselines import Baseline, BASELINES, get
from .cache import MaskCache
from .datasets import DavisLayout, GtCache, VideoSpec
from .handoff_metrics import score_curve
from .metrics import jf
from .protocol import SwitchManifest, SwitchPoint
from .results import RunRecord

__all__ = ["SoloRun", "solo_run", "evaluate_switch_point", "evaluate_dataset"]


@dataclass
class SoloRun:
    """One model's uninterrupted pass over one video, plus the states it shed."""

    model_id: str
    video: str
    masks: dict[int, Any]                                  # every frame produced
    states: dict[int, Any] = field(default_factory=dict)   # switch frame -> state
    from_cache: bool = False


def solo_run(predictor, spec: VideoSpec, first_prompt, export_at=()) -> SoloRun:
    """Prompt on frame 0, propagate to the end, snapshotting state at each of
    `export_at` as the cursor passes it."""
    video_dir = str(spec.frames[0].parent)
    session = predictor.init_video(video_dir, spec.obj_ids)

    frame0 = {}
    for oid, mask in first_prompt.items():
        frame0 = predictor.add_prompt(session, 0, oid, mask=mask)
    masks: dict[int, Any] = {0: dict(frame0)} if frame0 else {}

    cursor = 0
    states: dict[int, Any] = {}
    for f in sorted({int(x) for x in export_at}):
        if f > cursor:
            masks.update(predictor.run_until(session, f))
            cursor = f
        states[f] = predictor.export_state(session)

    last = predictor.frame_count(session) - 1
    if cursor < last:
        masks.update(predictor.continue_from(session, cursor + 1))
    return SoloRun(predictor.model_id, spec.name, masks, states)


class _SoloStore:
    """Solo runs for the video currently being evaluated.

    Scoped to one video deliberately: keeping every video's masks resident
    would grow without bound across a 30-video split, and nothing outside a
    video needs them.
    """

    def __init__(self, cache: MaskCache | None = None, fingerprints: dict | None = None):
        self.cache = cache or MaskCache(None)
        self.fingerprints = fingerprints or {}
        self._video: str | None = None
        self._runs: dict[str, SoloRun] = {}

    def for_video(self, video: str) -> None:
        if self._video != video:
            self._video, self._runs = video, {}

    def get(self, predictor, spec: VideoSpec, first_prompt, export_at) -> SoloRun:
        self.for_video(spec.name)
        have = self._runs.get(predictor.model_id)
        if have is not None and set(export_at) <= set(have.states):
            return have

        fp = self.fingerprints.get(predictor.model_id, "-")
        cached = self.cache.load(predictor.model_id, fp, spec.name, spec.num_frames)
        if cached is not None and not export_at:
            run = SoloRun(predictor.model_id, spec.name, cached, {}, from_cache=True)
        else:
            run = solo_run(predictor, spec, first_prompt, export_at)
            self.cache.save(predictor.model_id, fp, spec.name, spec.num_frames, run.masks)
        self._runs[predictor.model_id] = run
        return run


def _curve(masks, gt: GtCache, switch: int, obj_id: int, num_frames: int):
    """Per-frame J&F for one object, padded to full video length.

    nan outside the post-switch window, so `handoff_metrics` can index by
    absolute frame number and an unmeasured frame stays visibly unmeasured.
    """
    ms = np.full(num_frames, np.nan)
    for f in masks:
        if f <= switch or f >= num_frames:
            continue
        g = gt.mask(f, obj_id)
        pred = masks[f].get(obj_id)
        if pred is None:
            pred = np.zeros_like(g)
        ms[f] = jf(pred, g)[2]
    return ms


def _visibility(gt: GtCache, obj_id: int, num_frames: int) -> np.ndarray:
    """True on frames where this object's ground truth is non-empty."""
    return np.array([gt.visible(f, obj_id) for f in range(num_frames)], dtype=bool)


def evaluate_switch_point(
    *,
    dataset: DavisLayout,
    spec: VideoSpec,
    point: SwitchPoint,
    manifest: SwitchManifest,
    source,
    target,
    methods: list[str] | None = None,
    extra: list[Baseline] | None = None,
    gt: GtCache | None = None,
    store: _SoloStore | None = None,
    warm_curves: dict[int, np.ndarray] | None = None,
    visible: dict[int, np.ndarray] | None = None,
    export_at=(),
) -> list[RunRecord]:
    switch = point.frame
    gt = gt or GtCache(spec)
    store = store or _SoloStore()
    first_prompt = gt.masks(0)          # the one and only place GT acts as a prompt
    obj_ids = spec.obj_ids
    n = spec.num_frames

    src_solo = store.get(source, spec, first_prompt, export_at or (switch,))
    tgt_solo = store.get(target, spec, first_prompt, ())

    if visible is None:
        visible = {oid: _visibility(gt, oid, n) for oid in obj_ids}
    if warm_curves is None:
        warm_curves = {oid: _curve(tgt_solo.masks, gt, -1, oid, n) for oid in obj_ids}

    shared = dict(
        video=str(spec.frames[0].parent),
        obj_ids=obj_ids,
        first_prompt=first_prompt,
        source_masks={f: m for f, m in src_solo.masks.items() if f <= switch},
        source_state=src_solo.states[switch],
        source_solo=src_solo,
        target_solo=tgt_solo,
        switch=switch,
        source=source,
        target=target,
    )

    todo: list[Baseline] = [get(m) for m in (methods or list(BASELINES))]
    todo += extra or []

    records: list[RunRecord] = []
    for baseline in todo:
        result = baseline.run(**shared)
        # Post-switch frames the target processed and was scored on.  Zero for
        # source_only, where the source finished the video instead.
        target_frames = (sum(1 for f in result.masks if switch < f < n)
                         if result.by_target else 0)
        for oid in obj_ids:
            curve = _curve(result.masks, gt, switch, oid, n)
            scores = score_curve(curve, warm_curves[oid], switch, visible[oid])
            records.append(
                RunRecord(
                    dataset=dataset.name,
                    video=spec.name,
                    obj_id=oid,
                    fraction=point.fraction,
                    switch_frame=switch,
                    num_frames=n,
                    method=baseline.name,
                    source_model=source.model_id,
                    target_model=target.model_id,
                    manifest_digest=manifest.digest,
                    jf_gtvis=scores.jf_gtvis,
                    jf_at_5_gtvis=scores.jf_at_5_gtvis,
                    warm_jf_gtvis=scores.warm_jf_gtvis,
                    target_frames=target_frames,
                    notes=dict(result.notes),
                )
            )
    return records


def evaluate_dataset(
    *,
    dataset: DavisLayout,
    manifest: SwitchManifest,
    source,
    target,
    methods: list[str] | None = None,
    extra: list[Baseline] | None = None,
    limit: int | None = None,
    cache_dir: str | None = None,
    fingerprints: dict | None = None,
    on_video=None,
) -> list[RunRecord]:
    """Evaluate the source -> target handoff over the dataset."""
    out: list[RunRecord] = []
    store = _SoloStore(MaskCache(cache_dir), fingerprints)
    names = dataset.video_names[:limit] if limit else dataset.video_names

    for name in names:
        spec = dataset.video(name)
        points = manifest.for_video(name)
        if not points:
            continue

        gt = GtCache(spec)                    # decoded once, reused by every method
        switch_frames = tuple(p.frame for p in points)
        first_prompt = gt.masks(0)
        store.for_video(name)

        # One solo run per model, the source's state exported at every switch
        # frame, then scored once.
        store.get(source, spec, first_prompt, switch_frames)
        tgt_solo = store.get(target, spec, first_prompt, ())
        visible = {oid: _visibility(gt, oid, spec.num_frames) for oid in spec.obj_ids}
        warm = {oid: _curve(tgt_solo.masks, gt, -1, oid, spec.num_frames)
                for oid in spec.obj_ids}

        for point in points:
            out += evaluate_switch_point(
                dataset=dataset, spec=spec, point=point, manifest=manifest,
                source=source, target=target, methods=methods, extra=extra,
                gt=gt, store=store, warm_curves=warm,
                visible=visible, export_at=switch_frames,
            )
        if on_video:
            on_video(name, len(out))
    return out
