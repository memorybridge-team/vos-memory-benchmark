"""The contract between track A (infrastructure) and track B (evaluation).

Track B never imports SAM 2.  It drives whatever object satisfies `Predictor`,
so the same baselines run against SAM2-S, SAM2-L, Cutie or XMem once track D
wraps them.  Track A owns the implementation; this file owns the shape.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

# Per-frame binary masks keyed by frame index, one entry per tracked object id.
MaskSet = Mapping[int, np.ndarray]


@runtime_checkable
class SessionState(Protocol):
    """Opaque snapshot of everything a model accumulated up to some frame.

    For SAM 2 this wraps maskmem_features, maskmem_pos_enc, obj_ptr, the
    prompted-frame memory, and the per-frame occlusion flags.  Track B only
    requires that it round-trips through export/import and is picklable so runs
    can be cached.
    """

    frame_idx: int
    model_id: str


@runtime_checkable
class Predictor(Protocol):
    """A stateful video segmentation model, as track B needs to drive it."""

    model_id: str

    def init_video(self, video_dir: str, obj_ids: Iterable[int]) -> Any:
        """Open a video and return a fresh session with empty memory."""

    def add_prompt(
        self,
        session: Any,
        frame_idx: int,
        obj_id: int,
        *,
        mask: np.ndarray | None = None,
        points: np.ndarray | None = None,
        box: np.ndarray | None = None,
    ) -> MaskSet:
        """Prompt on one frame; returns that frame's prediction."""

    def seek(self, session: Any, frame_idx: int) -> None:
        """Move the session's read position to `frame_idx` WITHOUT processing
        any frame and WITHOUT touching memory.

        This is what makes replay-k mean "re-watch k frames" rather than "watch
        everything from the start", and what lets a transferred state resume at
        an arbitrary frame.  Implementations must not run the image encoder here.
        """

    def run_until(self, session: Any, frame_idx: int) -> dict[int, MaskSet]:
        """Propagate from the session's current position through `frame_idx`
        inclusive, accumulating memory.  Returns predictions per frame."""

    def continue_from(self, session: Any, frame_idx: int) -> dict[int, MaskSet]:
        """Propagate from `frame_idx` to the end of the video without
        re-processing anything before it."""

    def encode_memory(self, session: Any, frame_idx: int, masks: MaskSet) -> None:
        """Write one frame into the recent memory bank from masks that are
        *given*, using this model's own image and memory encoders.

        No prediction and no memory read happen: the frame's memory is a
        function of (image, given mask) only.  Costs exactly one image-encoder
        pass.  This is what `re_encode_oracle` is built from -- the target
        re-encodes the frames the source remembered, with the masks the source
        predicted, so its memory holds the source's information in the target's
        own representation.  All objects must already be registered, i.e. the
        frame-0 prompt must have been added first.
        """

    def export_state(self, session: Any) -> SessionState:
        """Snapshot the accumulated temporal memory."""

    def import_state(self, session: Any, state: SessionState) -> None:
        """Install a snapshot, replacing whatever memory the session held.

        Must accept a state whose `model_id` differs from this predictor's --
        that is the whole point of the study.  Raising on a mismatch is the
        job of a translator, not of the predictor.
        """

    def frame_count(self, session: Any) -> int:
        ...


@runtime_checkable
class Translator(Protocol):
    """Track C's object: maps one model's state into another's space."""

    name: str
    source_model: str
    target_model: str

    def translate(self, state: SessionState) -> SessionState:
        ...
