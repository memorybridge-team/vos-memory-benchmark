"""Result schema.

One row per (video, switch fraction, method, object).  Aggregation happens at
analysis time, never at write time -- a row that has been averaged away cannot
be un-averaged when a reviewer asks for the per-object breakdown.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pandas as pd

__all__ = ["RunRecord", "save_records", "load_records", "to_frame", "summarize",
           "retention"]


@dataclass
class RunRecord:
    dataset: str
    video: str
    obj_id: int
    fraction: float
    switch_frame: int
    num_frames: int
    method: str
    source_model: str
    target_model: str
    manifest_digest: str

    # --- the four metrics (see docs/METRICS.md) ----------------------------
    jf_gtvis: float          # GT-visible J&F over all post-switch frames
    jf_at_5_gtvis: float     # GT-visible J&F over the 5 frames after the switch
    warm_jf_gtvis: float     # Warm Target's jf_gtvis on the same frames (Retention)
    target_frames: int       # post-switch frames the target processed and was scored on

    notes: dict = field(default_factory=dict)


def to_frame(records: list[RunRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in records])


def save_records(records: list[RunRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        df = to_frame(records)
        df["notes"] = df["notes"].map(json.dumps)
        df.to_csv(path, index=False)
    else:
        path.write_text(
            json.dumps([asdict(r) for r in records], indent=2), encoding="utf-8"
        )


def load_records(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        df["notes"] = df["notes"].map(json.loads)
        return df
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def retention(df: pd.DataFrame, by: tuple[str, ...] = ("method",)) -> pd.DataFrame:
    """Retention = mean(jf_gtvis) / mean(warm_jf_gtvis) x 100, per group.

    A ratio of means, never a mean of ratios: dividing per row lets a video
    where the warm run itself scored 0.03 contribute a 2000%.  Both means are
    taken over the *same* rows, so a method cannot look good by having been
    scored on an easier subset.  Values above 100 are not clipped.
    """
    d = df.dropna(subset=["jf_gtvis", "warm_jf_gtvis"])
    out = d.groupby(list(by)).agg(
        jf_gtvis=("jf_gtvis", "mean"),
        warm_jf_gtvis=("warm_jf_gtvis", "mean"),
    )
    out["retention"] = 100.0 * out["jf_gtvis"] / out["warm_jf_gtvis"]
    return out.reset_index()


def summarize(df: pd.DataFrame, by: tuple[str, ...] = ("method",)) -> pd.DataFrame:
    """The four metrics, averaged over objects and videos.

    A nan `jf_at_5_gtvis` (no visible ground truth in the window) is left out
    of the mean, never treated as 0.
    """
    keys = list(by)
    agg = df.groupby(keys).agg(
        n=("jf_gtvis", "size"),
        jf_gtvis=("jf_gtvis", "mean"),
        jf_at_5_gtvis=("jf_at_5_gtvis", "mean"),
        target_frames=("target_frames", "mean"),
    )
    ret = retention(df, by).set_index(keys)[["retention"]]
    return agg.join(ret).reset_index()
