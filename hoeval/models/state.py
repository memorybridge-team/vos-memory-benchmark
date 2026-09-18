"""The portable form of a model's temporal memory.

A predictor exports one of these, a translator rewrites it, another predictor
imports it.  Nothing outside `models/` knows what SAM 2's `inference_state`
looks like, which is what lets track C write a translator against 64-channel
tensors instead of against a dict of dicts.

Layout
------
One `MemorySlot` per (object, frame) the model remembers, tagged by `kind`:

    prompted   the memory built on a frame that received a prompt (SAM 2's
               `cond_frame_outputs`).  Never evicted.
    recent     the rolling bank of the last N propagated frames (SAM 2 keeps
               N=6).  Evicted oldest-first.

The split matters for more than bookkeeping: track C fits a separate map per
kind, because the two distributions differ (a prompted memory encodes a mask the
model was handed, a recent memory encodes one it produced) and a single map
fitted across both would be a compromise between them.

What travels, and what does not
-------------------------------
Travelling: `maskmem` (the memory-encoder feature map), `obj_ptr`, and
`object_score_logits`.  The last one is the occlusion state.  Drop it and the
target treats an object the source had already lost as plainly visible, which
does not crash and does not show up in a roundtrip test -- it shows up as a
mysterious 10-point drop on the occlusion-heavy videos only.

Not travelling: spatial and temporal position encodings.  `import_state` has
the *target* re-issue those under its own convention.  Carrying the source's
would inject the source's coordinate frame into the target's attention -- the
same reason the cross-model KV cache work (arXiv 2608.03893) strips RoPE before
fitting a map and puts the target's back afterwards.

`extras` carries every other key the source predictor stored, verbatim and
untouched by translators.  It exists so that an exact same-model roundtrip stays
exact even for fields this module has never heard of; a translator that starts
depending on one should promote it to a real field first.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable

__all__ = [
    "SLOT_PROMPTED",
    "SLOT_RECENT",
    "SLOT_KINDS",
    "MemorySlot",
    "HandoffState",
]

SLOT_PROMPTED = "prompted"
SLOT_RECENT = "recent"
SLOT_KINDS = (SLOT_PROMPTED, SLOT_RECENT)


@dataclass
class MemorySlot:
    obj_id: int
    frame_idx: int
    kind: str                       # one of SLOT_KINDS
    maskmem: Any                    # (C, H, W), CPU; SAM 2: (64, 64, 64)
    obj_ptr: Any                    # (D,), CPU; SAM 2: (256,)
    object_score_logits: Any        # scalar; > 0 means "object is present"
    extras: dict[str, Any] = field(default_factory=dict)

    def replace_tensors(self, maskmem=None, obj_ptr=None) -> "MemorySlot":
        return replace(
            self,
            maskmem=self.maskmem if maskmem is None else maskmem,
            obj_ptr=self.obj_ptr if obj_ptr is None else obj_ptr,
        )


@dataclass
class HandoffState:
    model_id: str
    frame_idx: int                  # last frame the exporting model consumed
    obj_ids: tuple[int, ...]
    slots: list[MemorySlot] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- views -------------------------------------------------------------
    def of_kind(self, kind: str) -> list[MemorySlot]:
        return [s for s in self.slots if s.kind == kind]

    def for_object(self, obj_id: int) -> list[MemorySlot]:
        return [s for s in self.slots if s.obj_id == obj_id]

    def recent(self, obj_id: int) -> list[MemorySlot]:
        """Recent slots for one object, oldest first."""
        return sorted(
            (s for s in self.slots if s.obj_id == obj_id and s.kind == SLOT_RECENT),
            key=lambda s: s.frame_idx,
        )

    # -- edits (always return a new state; translators must not mutate) -----
    def with_slots(self, slots: Iterable[MemorySlot], **meta) -> "HandoffState":
        return HandoffState(
            model_id=self.model_id,
            frame_idx=self.frame_idx,
            obj_ids=self.obj_ids,
            slots=list(slots),
            meta={**self.meta, **meta},
        )

    def map_slots(self, fn: Callable[[MemorySlot], MemorySlot], **meta) -> "HandoffState":
        return self.with_slots([fn(s) for s in self.slots], **meta)

    def clone(self) -> "HandoffState":
        return copy.deepcopy(self)

    def describe(self) -> str:
        per_kind = {k: len(self.of_kind(k)) for k in SLOT_KINDS}
        return "HandoffState(%s @%d, objs=%s, slots=%s)" % (
            self.model_id, self.frame_idx, list(self.obj_ids), per_kind
        )
