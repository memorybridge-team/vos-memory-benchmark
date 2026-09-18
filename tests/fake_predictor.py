"""A stand-in for SAM 2 that has the same *state* behaviour without the weights.

Prediction quality is a deterministic function of what the session's memory
holds, so the baselines separate the way a memory-based model should: an anchor
memory helps, recent memories help more, a foreign state hurts unless it is
translated.  If a change to the harness breaks that ordering, the harness is
wrong -- that is what this fixture is for.

The representation gap, and what it is not
------------------------------------------
Each fake model encodes a frame the same way up to a per-model transform:

    maskmem = P_M . (b(frame) * sigma_M + mu_M)

where `b` is shared across models (the frame's actual content), `mu_M/sigma_M`
are the model's scale convention, and `P_M` is a fixed channel permutation
standing in for a genuine difference of representation.  A session measures how
much of an imported state it can actually read by inverting its *own* convention
and comparing to the truth, and its predictions degrade with that fidelity.

The consequence is that the three Layer 2 cases land in three different places:
Direct Copy is hurt by both `sigma/mu` and `P`; Norm-Matched fixes `sigma/mu`
and is still hurt by `P`; a map that can permute (Procrustes, track C) fixes
both.  That ordering is *built into the fixture*, so a test asserting it is
testing the plumbing -- that `NormMatched` actually reaches the tensors, that
a translated state is scored by the same code path -- and nothing whatever
about SAM 2.  Whether the real gap is scale or
representation is the experiment; this file must never be quoted as evidence
for either answer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoeval.models.state import HandoffState, MemorySlot, SLOT_PROMPTED, SLOT_RECENT

ANCHOR_WEIGHT = 0.45
RECENT_WEIGHT = 0.40
LASTMASK_WEIGHT = 0.35
N_RECENT = 6           # SAM 2 keeps 6 recent memories
LOST_BELOW = 0.25
CHANNELS = 16
GRID = 4
PTR_DIM = 8


def _model_convention(model_id: str):
    """A stable per-model (mu, sigma, channel permutation)."""
    seed = int(hashlib.sha256(model_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 2.0, size=CHANNELS)
    sigma = rng.uniform(0.5, 2.5, size=CHANNELS)
    # Only half the channels are shuffled.  A fully independent permutation
    # would leave Direct Copy and Norm-Matched both at chance and the fixture
    # could not tell a working translator from a broken one; a partial one
    # leaves a scale-fixable component and a structural one, which is the
    # situation the real study is trying to distinguish between.
    perm = np.arange(CHANNELS)
    half = CHANNELS // 2
    perm[:half] = rng.permutation(half)
    return mu, sigma, perm


def _content(frame_idx: int, obj_id: int) -> np.ndarray:
    """What the frame actually contains -- identical for every model."""
    rng = np.random.default_rng(1000 * obj_id + frame_idx)
    return rng.normal(0.0, 1.0, size=CHANNELS)


@dataclass
class FakeSession:
    video_dir: Path
    obj_ids: tuple[int, ...]
    n_frames: int
    pos: int = -1
    anchor: bool = False
    recent: int = 0
    last_prompt_frame: int | None = None
    fidelity: float = 1.0
    slots: list[MemorySlot] = field(default_factory=list)
    prompts: dict = field(default_factory=dict)


class FakePredictor:
    def __init__(self, model_id: str, skill: float = 1.0):
        self.model_id = model_id
        self.skill = skill  # a "large" model degrades more gracefully
        self.mu, self.sigma, self.perm = _model_convention(model_id)
        self.inv_perm = np.argsort(self.perm)

    # -- the model's private encoding of a frame ---------------------------
    def _encode(self, frame_idx: int, obj_id: int) -> np.ndarray:
        b = _content(frame_idx, obj_id)
        chan = b * self.sigma + self.mu
        spatial = np.tile(chan[:, None, None], (1, GRID, GRID))
        return spatial[self.perm]

    def _decode(self, maskmem: np.ndarray) -> np.ndarray:
        chan = np.asarray(maskmem)[self.inv_perm].mean(axis=(1, 2))
        return (chan - self.mu) / self.sigma

    def _fidelity(self, slots) -> float:
        """How much of an imported memory this model can actually read."""
        sims = []
        for s in slots:
            got = self._decode(s.maskmem)
            want = _content(s.frame_idx, s.obj_id)
            denom = np.linalg.norm(got) * np.linalg.norm(want)
            sims.append(0.0 if denom == 0 else float(np.dot(got, want) / denom))
        if not sims:
            return 1.0
        return float(np.clip(np.mean(sims), 0.0, 1.0))

    def _remember(self, session: FakeSession, frame_idx: int, kind: str) -> None:
        for oid in session.obj_ids:
            session.slots.append(
                MemorySlot(
                    obj_id=oid,
                    frame_idx=frame_idx,
                    kind=kind,
                    maskmem=self._encode(frame_idx, oid),
                    obj_ptr=self._encode(frame_idx, oid)[:PTR_DIM, 0, 0].copy(),
                    object_score_logits=1.0 if self._gt(session, frame_idx, oid).any() else -4.0,
                )
            )
        # the recent bank evicts oldest-first, exactly like SAM 2's
        for oid in session.obj_ids:
            rec = [s for s in session.slots if s.obj_id == oid and s.kind == SLOT_RECENT]
            for dead in sorted(rec, key=lambda s: s.frame_idx)[:-N_RECENT]:
                session.slots.remove(dead)
        # Fidelity is a property of what is in memory *now*, not of what was
        # imported once.  A session that took a foreign state and then kept
        # propagating flushes those slots out of the recent bank within N
        # frames and partially recovers -- which is what a real model does, and
        # which is why a bad handoff has to be measured near the switch (@5)
        # and not only as a full-video average.  A foreign *prompted* slot is
        # never evicted, so the anchor keeps hurting for the whole video.
        session.fidelity = self._fidelity(session.slots)

    # -- session -----------------------------------------------------------
    def init_video(self, video_dir, obj_ids):
        d = Path(video_dir)
        n = len(list(d.glob("*.jpg")))
        return FakeSession(d, tuple(obj_ids), n)

    def frame_count(self, session):
        return session.n_frames

    # -- ground truth the fake model "sees" --------------------------------
    def _gt(self, session, frame_idx, obj_id):
        ann = session.video_dir.parent.parent / "Annotations" / session.video_dir.name
        ids = np.array(Image.open(ann / ("%05d.png" % frame_idx)).convert("P"))
        return ids == obj_id

    def _quality(self, session, frame_idx):
        q = 0.0
        if session.anchor:
            q += ANCHOR_WEIGHT
        q += RECENT_WEIGHT * min(session.recent, N_RECENT) / N_RECENT
        if session.last_prompt_frame is not None:
            decay = max(0.0, 1.0 - (frame_idx - session.last_prompt_frame) / 25.0)
            q += LASTMASK_WEIGHT * decay
        # Complete memory (anchor + a full recent queue) is the best this fake
        # model can do; a prompt cannot push it past that.  Real models are not
        # bound by this -- a mid-video prompt injects information the warm run
        # never had -- which is why the harness does not treat Warm Target as a
        # hard ceiling, only as the reference the gap is measured against.
        q = min(q, ANCHOR_WEIGHT + RECENT_WEIGHT)
        return min(1.0, q * session.fidelity * self.skill)

    def _predict(self, session, frame_idx):
        q = self._quality(session, frame_idx)
        out = {}
        for oid in session.obj_ids:
            gt = self._gt(session, frame_idx, oid)
            if q < LOST_BELOW:
                out[oid] = np.zeros_like(gt)
                continue
            shift = int(round((1.0 - q) * 12))
            out[oid] = np.roll(gt, shift, axis=0) if shift else gt.copy()
        return out

    # -- Predictor protocol ------------------------------------------------
    def add_prompt(self, session, frame_idx, obj_id, *, mask=None, points=None, box=None):
        if frame_idx == 0:
            session.anchor = True
        else:
            session.last_prompt_frame = frame_idx
        session.prompts[(frame_idx, obj_id)] = mask
        session.pos = max(session.pos, frame_idx)
        if not any(s.frame_idx == frame_idx and s.kind == SLOT_PROMPTED for s in session.slots):
            self._remember(session, frame_idx, SLOT_PROMPTED)
        return self._predict(session, frame_idx)

    def seek(self, session, frame_idx):
        session.pos = frame_idx - 1

    def run_until(self, session, frame_idx):
        preds = {}
        start = max(0, session.pos + 1)
        for f in range(start, frame_idx + 1):
            preds[f] = self._predict(session, f)
            session.recent += 1
            self._remember(session, f, SLOT_RECENT)
        session.pos = frame_idx
        return preds

    def continue_from(self, session, frame_idx):
        session.pos = frame_idx - 1
        preds = {}
        for f in range(frame_idx, session.n_frames):
            preds[f] = self._predict(session, f)
            session.recent += 1
            self._remember(session, f, SLOT_RECENT)
        session.pos = session.n_frames - 1
        return preds

    def encode_memory(self, session, frame_idx, masks):
        # One frame, encoded under this model's own convention: the slot it
        # writes is native whatever model produced the mask.  The fake ignores
        # the mask content, as it does everywhere -- quality is a function of
        # what memory holds.
        session.recent += 1
        self._remember(session, frame_idx, SLOT_RECENT)
        session.pos = max(session.pos, frame_idx)

    def export_state(self, session):
        return HandoffState(
            model_id=self.model_id,
            frame_idx=session.pos,
            obj_ids=session.obj_ids,
            slots=[MemorySlot(s.obj_id, s.frame_idx, s.kind, np.array(s.maskmem),
                              np.array(s.obj_ptr), s.object_score_logits)
                   for s in session.slots],
            meta={"last_prompt_frame": session.last_prompt_frame},
        )

    def import_state(self, session, state):
        session.slots = [MemorySlot(s.obj_id, s.frame_idx, s.kind, np.array(s.maskmem),
                                    np.array(s.obj_ptr), s.object_score_logits)
                         for s in state.slots]
        session.anchor = any(s.kind == SLOT_PROMPTED for s in session.slots)
        per_obj = [len(state.recent(o)) for o in state.obj_ids] or [0]
        session.recent = min(N_RECENT, max(per_obj))
        session.last_prompt_frame = state.meta.get("last_prompt_frame")
        session.pos = state.frame_idx
        # A state this model cannot read is not an error -- it is a bad handoff,
        # and the whole study is about how bad.  Never raise on model mismatch.
        session.fidelity = self._fidelity(session.slots)
