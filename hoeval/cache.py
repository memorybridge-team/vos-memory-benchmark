"""On-disk cache for a model's solo run over a video.

Only masks are cached.  They are the expensive-to-recompute, cheap-to-store
half: bit-packed, one DAVIS video of predictions is a few megabytes, while the
exported states are ~1 MB per memory slot and are only ever reused inside the
video that produced them -- which the in-process cache already covers.

The cache key includes the model id, the checkpoint fingerprint and the frame
count, so a re-downloaded or swapped checkpoint can never be served stale masks
from an earlier one.  That is not paranoia: SAM 2 and SAM 2.1 share file names
and differ in memory behaviour, and a silently-mixed pair would produce numbers
that look plausible and mean nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["MaskCache"]


class MaskCache:
    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, model_id: str, fingerprint: str, video: str, n_frames: int) -> Path:
        key = hashlib.sha256(
            "|".join([model_id, fingerprint, video, str(n_frames)]).encode()
        ).hexdigest()[:20]
        return self.root / ("%s__%s__%s.npz" % (model_id, video, key))

    def load(self, model_id, fingerprint, video, n_frames):
        if not self.root:
            return None
        p = self._path(model_id, fingerprint, video, n_frames)
        if not p.exists():
            return None
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            packed, shape = z["packed"], tuple(z["shape"])
            flat = np.unpackbits(packed, count=int(np.prod(shape))).astype(bool)
            stacked = flat.reshape(shape)
        frames, obj_ids = meta["frames"], meta["obj_ids"]
        return {
            f: {oid: stacked[i, j] for j, oid in enumerate(obj_ids)}
            for i, f in enumerate(frames)
        }

    def save(self, model_id, fingerprint, video, n_frames, masks) -> None:
        if not self.root or not masks:
            return
        frames = sorted(masks)
        obj_ids = sorted(masks[frames[0]])
        stacked = np.stack(
            [np.stack([np.asarray(masks[f][o], dtype=bool) for o in obj_ids])
             for f in frames]
        )
        p = self._path(model_id, fingerprint, video, n_frames)
        np.savez_compressed(
            p,
            packed=np.packbits(stacked.reshape(-1)),
            shape=np.asarray(stacked.shape),
            meta=json.dumps({"frames": frames, "obj_ids": obj_ids}),
        )
