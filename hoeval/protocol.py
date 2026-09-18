"""Switch-point protocol.

Every method in the study must be evaluated at *byte-identical* switch points,
otherwise a comparison between Last Mask, Replay-k and a translator is
meaningless.  We therefore resolve switch points once, freeze them into a
manifest, hash the manifest, and have every run assert against that hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

__all__ = ["SwitchPoint", "SwitchManifest", "build_manifest", "load_manifest"]

DEFAULT_FRACTIONS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class SwitchPoint:
    video: str
    num_frames: int
    fraction: float
    frame: int  # source runs [0..frame], target continues from frame+1

    @property
    def key(self) -> str:
        return f"{self.video}@{self.fraction:.2f}"


def resolve_frame(num_frames: int, fraction: float) -> int:
    """Frame index the source model stops after.

    Clamped so that at least one frame precedes the switch (the prompted first
    frame) and at least one frame follows it (otherwise there is nothing to
    score post-switch).
    """
    if num_frames < 3:
        raise ValueError(f"video too short to switch: {num_frames} frames")
    frame = int(round(fraction * (num_frames - 1)))
    return max(1, min(frame, num_frames - 2))


@dataclass
class SwitchManifest:
    dataset: str
    fractions: tuple[float, ...]
    points: list[SwitchPoint]
    digest: str

    def for_video(self, video: str) -> list[SwitchPoint]:
        return [p for p in self.points if p.video == video]

    def assert_matches(self, other_digest: str) -> None:
        if self.digest != other_digest:
            raise RuntimeError(
                "switch manifest mismatch: results were produced against a "
                f"different protocol ({other_digest} != {self.digest})"
            )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "fractions": list(self.fractions),
            "points": [asdict(p) for p in self.points],
            "digest": self.digest,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _digest(points: list[SwitchPoint]) -> str:
    blob = "\n".join(f"{p.video}|{p.num_frames}|{p.fraction:.4f}|{p.frame}" for p in points)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_manifest(
    dataset: str,
    video_lengths: dict[str, int],
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> SwitchManifest:
    """Deterministic: same video lengths in, same manifest out."""
    points: list[SwitchPoint] = []
    for video in sorted(video_lengths):
        n = video_lengths[video]
        if n < 3:
            continue  # skipped videos are recorded by their absence; see report
        for frac in fractions:
            points.append(SwitchPoint(video, n, float(frac), resolve_frame(n, frac)))
    return SwitchManifest(dataset, tuple(fractions), points, _digest(points))


def load_manifest(path: str | Path) -> SwitchManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    points = [SwitchPoint(**p) for p in payload["points"]]
    manifest = SwitchManifest(
        payload["dataset"], tuple(payload["fractions"]), points, payload["digest"]
    )
    if _digest(points) != manifest.digest:
        raise RuntimeError(f"manifest {path} is corrupt: digest does not match its points")
    return manifest
