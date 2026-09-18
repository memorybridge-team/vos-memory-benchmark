"""Named model configurations, so a run is reproducible from its command line.

Every record carries the checkpoint fingerprint these produce.  That is not
bureaucracy: SAM 2 and SAM 2.1 ship checkpoints with interchangeable names and
*different memory behaviour*, so a table built from a mixed pair is wrong in a
way no metric will flag.  Pin the variant, record the fingerprint, and a
reviewer can tell which one produced a number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ModelSpec", "MODELS", "resolve", "build"]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    config: str
    checkpoint: str     # relative to the sam2 checkpoints directory
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "sam2_small": ModelSpec(
        "sam2_small", "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt", "SAM 2.1 Hiera-S",
    ),
    "sam2_large": ModelSpec(
        "sam2_large", "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt", "SAM 2.1 Hiera-L",
    ),
    "sam2_tiny": ModelSpec(
        "sam2_tiny", "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt", "SAM 2.1 Hiera-T",
    ),
    "sam2_base_plus": ModelSpec(
        "sam2_base_plus", "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt", "SAM 2.1 Hiera-B+",
    ),
}


def resolve(name: str, checkpoints_dir: str | os.PathLike) -> tuple[ModelSpec, Path]:
    if name not in MODELS:
        raise KeyError("unknown model %r; have %s" % (name, sorted(MODELS)))
    spec = MODELS[name]
    path = Path(checkpoints_dir) / spec.checkpoint
    if not path.exists():
        raise FileNotFoundError(
            "%s not found -- run checkpoints/download_ckpts.sh in the sam2 "
            "checkout, and make sure it is the 2.1 set: mixing 2.0 and 2.1 "
            "silently changes memory behaviour." % path
        )
    return spec, path


def build(name: str, checkpoints_dir, device: str = "cuda", **kw):
    """Import-on-call so that nothing here drags torch into a laptop session."""
    from .sam2_wrapper import Sam2Predictor  # noqa: PLC0415

    spec, path = resolve(name, checkpoints_dir)
    return Sam2Predictor(spec.model_id, spec.config, str(path), device=device, **kw)
