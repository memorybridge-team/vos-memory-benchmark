"""SAM 2 behind the `Predictor` protocol.

This is the only file in the project that imports SAM 2.  Everything else --
baselines, translators, metrics -- works against `Predictor` and `HandoffState`,
which is what lets the same study run against Cutie or XMem later without
touching the harness.

The four operations that are easy to get subtly wrong
-----------------------------------------------------
**seek** moves the cursor and runs nothing.  If it quietly re-encoded frames,
`replay_k` would stop meaning "re-watch k frames" and the entire replay column
would be measuring the wrong thing while still looking sensible.

**export_state** carries `object_score_logits`.  It is the occlusion state, it
is one line to forget, and forgetting it does not crash and does not fail a
roundtrip test -- it shows up as an unexplained loss on exactly the videos with
heavy occlusion, which are the ones the study cares most about.

**export_state** also carries the object index mapping.  SAM 2 stores memory
under a dense internal index, not under your object id.  Hand memory across
without remapping and every tensor is numerically perfect and attached to the
wrong object.

**import_state** does not copy position encodings.  Spatial `maskmem_pos_enc`
is re-issued by the target, and SAM 2's temporal encoding is added at attention
time from the *current* frame distances, so it never travels.  Carrying the
source's would inject the source's coordinate frame into the target's attention
-- the same reason the cross-model KV cache work (arXiv 2608.03893) strips RoPE
before fitting a map and lets the target reapply its own.

How much memory is a "full" state
---------------------------------
SAM 2 keeps every frame it ever tracked in `output_dict`, but only *reads* a
bounded window: `num_maskmem - 1` recent memory slots, and (when object pointers
are enabled in the encoder) obj_ptrs from up to `max_obj_ptrs_in_encoder - 1`
recent frames -- 6 and 15 respectively at the stock settings.  So the working
set is the larger of the two, and that is what `export_state` ships.

Exporting fewer would break the roundtrip test in a way that looks like a
translator problem.  Exporting all of history would inflate every state file
with slots the model provably never reads.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .sam2_layout import MEMORY_KEYS, REGENERATED_KEYS, StateLayout, probe_layout
from .state import HandoffState, MemorySlot, SLOT_PROMPTED, SLOT_RECENT

__all__ = ["Sam2Predictor", "Sam2Session", "checkpoint_fingerprint"]


def checkpoint_fingerprint(path: str | Path) -> str:
    """First 16 hex of sha256 over the checkpoint's first and last megabyte.

    Full hashing of a 900 MB checkpoint on every run is wasteful; the ends are
    enough to catch the failure that actually happens, which is a SAM 2 and a
    SAM 2.1 checkpoint of the same name being mixed inside one results table.
    Recorded on every row.
    """
    p = Path(path)
    h = hashlib.sha256()
    size = p.stat().st_size
    with p.open("rb") as fh:
        h.update(fh.read(1 << 20))
        if size > (2 << 20):
            fh.seek(-(1 << 20), 2)
            h.update(fh.read(1 << 20))
    h.update(str(size).encode())
    return h.hexdigest()[:16]


@dataclass
class Sam2Session:
    inference_state: Any
    video_dir: str
    obj_ids: tuple[int, ...]
    layout: StateLayout
    cursor: int = -1            # last frame consumed; -1 = nothing yet
    prompted: set = field(default_factory=set)
    encoder_calls: int = 0      # image-encoder invocations, to audit seek


class Sam2Predictor:
    """Wraps `sam2.build_sam.build_sam2_video_predictor`."""

    def __init__(
        self,
        model_id: str,
        config: str,
        checkpoint: str,
        device: str = "cuda",
        recent_slots: int | None = None,
        mask_threshold: float = 0.0,
    ):
        import torch  # noqa: PLC0415
        from sam2.build_sam import build_sam2_video_predictor  # noqa: PLC0415

        self.model_id = model_id
        self.config = config
        self.checkpoint = str(checkpoint)
        self.fingerprint = checkpoint_fingerprint(checkpoint)
        self.device = device
        self.mask_threshold = mask_threshold
        self._torch = torch

        self.model = build_sam2_video_predictor(config, self.checkpoint, device=device)
        self.model.eval()

        # How far back the model actually reads.  Derived from the model, not
        # assumed: a build with different settings must change this number, not
        # silently ship a state that is missing what it reads.
        n_mem = int(getattr(self.model, "num_maskmem", 7)) - 1
        n_ptr = 0
        if getattr(self.model, "use_obj_ptrs_in_encoder", False):
            n_ptr = int(getattr(self.model, "max_obj_ptrs_in_encoder", 16)) - 1
        self.recent_slots = int(recent_slots or max(n_mem, n_ptr, 1))
        self.reads = {"num_maskmem": n_mem, "obj_ptr_window": n_ptr}

        self._wrap_encoder_counter()

    # ------------------------------------------------------------------ cost
    def _wrap_encoder_counter(self) -> None:
        """Count real image-encoder passes per session.

        Lets tests/test_state_roundtrip.py check that `seek` runs nothing: if it
        re-encoded frames, replay_k would quietly re-watch the whole video.
        """
        inner = self.model.forward_image
        holder = {"session": None}

        def counted(img):
            s = holder["session"]
            if s is not None:
                s.encoder_calls += 1
            return inner(img)

        self.model.forward_image = counted
        self._counter = holder

    def _bind(self, session: Sam2Session | None) -> None:
        self._counter["session"] = session

    # --------------------------------------------------------------- session
    def init_video(self, video_dir: str, obj_ids: Iterable[int]) -> Sam2Session:
        state = self.model.init_state(video_path=str(video_dir))
        layout = probe_layout(state).require()
        return Sam2Session(state, str(video_dir), tuple(int(o) for o in obj_ids), layout)

    def frame_count(self, session: Sam2Session) -> int:
        return int(session.inference_state["num_frames"])

    # ---------------------------------------------------------------- prompts
    def add_prompt(self, session, frame_idx, obj_id, *, mask=None, points=None, box=None):
        self._bind(session)
        try:
            if mask is not None:
                _, ids, logits = self.model.add_new_mask(
                    session.inference_state, frame_idx, int(obj_id),
                    self._torch.as_tensor(np.asarray(mask, dtype=bool)),
                )
            else:
                _, ids, logits = self.model.add_new_points_or_box(
                    session.inference_state, frame_idx, int(obj_id),
                    points=points, box=box,
                    labels=None if points is None else np.ones(len(points), dtype=np.int32),
                )
        finally:
            self._bind(None)
        session.prompted.add(int(frame_idx))
        session.cursor = max(session.cursor, int(frame_idx))
        return self._to_masks(ids, logits)

    # ------------------------------------------------------------ propagation
    def seek(self, session: Sam2Session, frame_idx: int) -> None:
        """Move the read position. Runs nothing, touches no memory."""
        session.cursor = int(frame_idx) - 1

    def _propagate(self, session: Sam2Session, start: int, stop: int) -> dict:
        """Frames [start, stop] inclusive. Empty dict when there is nothing to do."""
        if start > stop:
            return {}
        out: dict[int, dict] = {}
        self._bind(session)
        try:
            with self._torch.inference_mode():
                for f, ids, logits in self.model.propagate_in_video(
                    session.inference_state,
                    start_frame_idx=int(start),
                    max_frame_num_to_track=int(stop - start),
                ):
                    out[int(f)] = self._to_masks(ids, logits)
                    if int(f) >= stop:
                        break
        finally:
            self._bind(None)
        session.cursor = max(session.cursor, stop)
        return out

    def run_until(self, session: Sam2Session, frame_idx: int) -> dict:
        return self._propagate(session, session.cursor + 1, int(frame_idx))

    def continue_from(self, session: Sam2Session, frame_idx: int) -> dict:
        session.cursor = int(frame_idx) - 1
        return self._propagate(session, int(frame_idx), self.frame_count(session) - 1)

    # --------------------------------------------------------------- re-encode
    def encode_memory(self, session: Sam2Session, frame_idx: int, masks) -> None:
        """Write `frame_idx` into the recent bank from given masks, per object.

        Goes through SAM 2's own single-frame path with the mask as input and
        `is_init_cond_frame=False`.  With `use_mask_input_as_output_without_sam`
        (on in every released config) `_track_step` takes the mask-as-output
        branch *before* `_prepare_memory_conditioned_features`, so the result is
        a function of the image and the given mask only -- no memory is read and
        nothing is predicted.  The output lands where SAM 2 keeps a propagated
        frame (`non_cond_frame_outputs`), so the memory selector later reads it
        by frame distance exactly as it would a frame it tracked itself.

        One difference from a tracked frame: the stored mask logits are the
        binary input mapped to +-10 (`_use_mask_as_output`), not the decoder's
        soft logits, because the source's masks arrive binarised.  The memory
        encoder therefore sees sigmoid(+-10) ~ {0, 1} rather than a soft map.
        """
        import torch.nn.functional as F  # noqa: PLC0415

        torch = self._torch
        state = session.inference_state
        if not getattr(self.model, "use_mask_input_as_output_without_sam", False):
            raise RuntimeError(
                "this SAM 2 build runs the mask decoder on mask inputs, so "
                "encode_memory would read memory and predict; re_encode_oracle "
                "would no longer mean 'source masks, target encoders'."
            )
        id_to_idx = state["obj_id_to_idx"]
        size = int(self.model.image_size)
        f = int(frame_idx)

        self._bind(session)
        try:
            with torch.inference_mode():
                for oid in session.obj_ids:
                    obj_idx = id_to_idx.get(int(oid))
                    if obj_idx is None:
                        raise RuntimeError(
                            "object %d is not registered; add the frame-0 prompt "
                            "before encode_memory" % oid
                        )
                    mask = masks.get(int(oid))
                    if mask is None:
                        mask = self._empty(session)
                    m = torch.as_tensor(np.asarray(mask, dtype=bool))[None, None]
                    m = m.float().to(state["device"])
                    # Same resize as SAM 2's add_new_mask.
                    if tuple(m.shape[-2:]) != (size, size):
                        m = F.interpolate(m, size=(size, size), mode="bilinear",
                                          align_corners=False, antialias=True)
                        m = (m >= 0.5).float()
                    per = self._per_obj(session)[obj_idx]
                    # The image features are cached per frame, so objects after
                    # the first reuse them: one encoder pass per frame.
                    out, _ = self.model._run_single_frame_inference(
                        inference_state=state,
                        output_dict=per,
                        frame_idx=f,
                        batch_size=1,
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=m,
                        reverse=False,
                        run_mem_encoder=True,
                    )
                    per["non_cond_frame_outputs"][f] = out
                    if session.layout.frames_tracked_per_obj:
                        state["frames_tracked_per_obj"].setdefault(obj_idx, {})[f] = {
                            "reverse": False
                        }
        finally:
            self._bind(None)
        session.cursor = max(session.cursor, f)

    # ------------------------------------------------------------------ masks
    def _empty(self, session) -> np.ndarray:
        h = int(session.inference_state["video_height"])
        w = int(session.inference_state["video_width"])
        return np.zeros((h, w), dtype=bool)

    def _to_masks(self, obj_ids, logits) -> dict[int, np.ndarray]:
        arr = logits.detach().cpu().numpy()
        return {
            int(o): (arr[i, 0] > self.mask_threshold)
            for i, o in enumerate(list(obj_ids))
        }

    # ------------------------------------------------------------------ state
    def _per_obj(self, session) -> dict:
        return session.inference_state["output_dict_per_obj"]

    def _slot(self, obj_id, frame_idx, kind, out) -> MemorySlot:
        maskmem = out.get("maskmem_features")
        return MemorySlot(
            obj_id=int(obj_id),
            frame_idx=int(frame_idx),
            kind=kind,
            maskmem=None if maskmem is None else maskmem.detach().to("cpu").clone().squeeze(0),
            obj_ptr=out["obj_ptr"].detach().to("cpu").clone().squeeze(0),
            object_score_logits=out["object_score_logits"].detach().to("cpu").clone(),
            extras={
                k: (v.detach().to("cpu").clone() if hasattr(v, "detach") else copy.deepcopy(v))
                for k, v in out.items()
                if k not in MEMORY_KEYS and k not in REGENERATED_KEYS
            },
        )

    def export_state(self, session: Sam2Session) -> HandoffState:
        """Snapshot everything the model would read to predict `cursor + 1`."""
        idx_to_id = session.inference_state["obj_idx_to_id"]
        cursor = session.cursor
        slots: list[MemorySlot] = []
        for obj_idx, per in self._per_obj(session).items():
            obj_id = int(idx_to_id[obj_idx])
            for f, out in sorted(per.get("cond_frame_outputs", {}).items()):
                if f <= cursor:
                    slots.append(self._slot(obj_id, f, SLOT_PROMPTED, out))
            recent = sorted(f for f in per.get("non_cond_frame_outputs", {}) if f <= cursor)
            for f in recent[-self.recent_slots:]:
                slots.append(self._slot(obj_id, f, SLOT_RECENT,
                                        per["non_cond_frame_outputs"][f]))
        return HandoffState(
            model_id=self.model_id,
            frame_idx=cursor,
            obj_ids=session.obj_ids,
            slots=slots,
            meta={
                "checkpoint": self.checkpoint,
                "fingerprint": self.fingerprint,
                "recent_slots": self.recent_slots,
                "reads": dict(self.reads),
                "prompted_frames": sorted(session.prompted),
                "video": session.video_dir,
            },
        )

    # -- import ------------------------------------------------------------
    def _pos_enc(self, session, like):
        """The *target's* spatial position encoding for a memory map.

        Regenerated, never imported.  SAM 2 derives this from a sinusoidal
        position encoding of the feature grid rather than learning it, so
        producing it here is exact and costs no forward pass.  If a session has
        already computed one, reuse that.
        """
        cached = session.inference_state.get("constants", {}).get("maskmem_pos_enc")
        if cached is not None:
            return [c.clone() for c in cached] if isinstance(cached, list) else cached.clone()
        enc = self.model.memory_encoder.position_encoding(like).to(like.dtype)
        return [enc]

    def import_state(self, session: Sam2Session, state: HandoffState) -> None:
        """Install a foreign (or native) state. Never raises on model mismatch --
        judging a bad handoff is the study's job, not the wrapper's."""
        dev = session.inference_state["device"]
        store = session.inference_state.get("storage_device", dev)

        # Register every object first: SAM 2 assigns dense internal indices and
        # refuses new objects once tracking has started, so this must happen
        # before anything else touches the session.
        for obj_id in state.obj_ids:
            self.model._obj_id_to_idx(session.inference_state, int(obj_id))
        id_to_idx = session.inference_state["obj_id_to_idx"]

        for slot in state.slots:
            obj_idx = id_to_idx[int(slot.obj_id)]
            per = self._per_obj(session)[obj_idx]
            bucket = ("cond_frame_outputs" if slot.kind == SLOT_PROMPTED
                      else "non_cond_frame_outputs")
            out = {k: (v.clone() if hasattr(v, "clone") else copy.deepcopy(v))
                   for k, v in slot.extras.items()}
            mm = slot.maskmem
            if mm is not None:
                mm = self._torch.as_tensor(mm).to(store).unsqueeze(0)
            out["maskmem_features"] = mm
            out["maskmem_pos_enc"] = None if mm is None else self._pos_enc(session, mm)
            out["obj_ptr"] = self._torch.as_tensor(slot.obj_ptr).to(dev).unsqueeze(0)
            out["object_score_logits"] = self._torch.as_tensor(
                slot.object_score_logits
            ).reshape(1, 1).to(dev)
            for key, value in list(out.items()):
                if hasattr(value, "to") and key not in ("maskmem_features", "maskmem_pos_enc"):
                    out[key] = value.to(dev)
            per[bucket][int(slot.frame_idx)] = out

            if session.layout.frames_tracked_per_obj:
                session.inference_state["frames_tracked_per_obj"].setdefault(
                    obj_idx, {}
                )[int(slot.frame_idx)] = {"reverse": False}

        session.prompted = {s.frame_idx for s in state.slots if s.kind == SLOT_PROMPTED}
        session.cursor = int(state.frame_idx)
