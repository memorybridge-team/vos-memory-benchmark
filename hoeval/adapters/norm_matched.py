"""Per-channel distribution matching -- the cheapest thing that is not nothing.

    x' = (x - mu_S) / sigma_S * sigma_T + mu_T

mu_S and mu_T are measured *independently*, each from its own model's run over
the DAVIS train split.  No pairing, no correspondence, no shared video even --
which is the only reason this can run before the paired-state pipeline exists.
It is the null hypothesis for every fitted translator: if a per-channel affine
rescale recovers most of what Direct Copy loses, then the gap was calibration,
not representation, and a learned map has to justify itself against that.

What the outcome means, stated before the numbers arrive so it cannot be
rationalised afterwards:

    Direct fails, norm_matched fails   -> the spaces differ in more than scale;
                                          a translator is warranted.
    Direct fails, norm_matched works   -> it was calibration.  The paper's
                                          story becomes "distribution matching
                                          plus alpha", and the translator has to
                                          beat this line, not Direct Copy.

Statistics come from the **train** split.  The val videos this is evaluated on
must never touch the fit -- including through a statistic as innocent as a
channel mean.
"""

from __future__ import annotations

from ..models.state import HandoffState, MemorySlot
from ._array import like, to_numpy
from .stats import StatsBook, group_of

__all__ = ["NormMatched"]


class NormMatched:
    """Satisfies `interfaces.Translator`."""

    name = "norm_matched"

    def __init__(self, source_stats: StatsBook, target_stats: StatsBook):
        self.source_stats = source_stats
        self.target_stats = target_stats
        self.source_model = source_stats.model_id
        self.target_model = target_stats.model_id
        if source_stats.split != "train" or target_stats.split != "train":
            raise ValueError(
                "norm_matched statistics must come from the train split "
                "(got %r / %r) -- fitting on val leaks the evaluation set"
                % (source_stats.split, target_stats.split)
            )

    def _match(self, arr, group: str):
        s = self.source_stats.get(group)
        t = self.target_stats.get(group)
        x = to_numpy(arr)
        if x.shape[0] != s.mean.shape[0]:
            raise ValueError(
                "%s: statistics have %d channels, tensor has %d"
                % (group, s.mean.shape[0], x.shape[0])
            )
        # broadcast the per-channel vectors over whatever trails the channel axis
        shape = (-1,) + (1,) * (x.ndim - 1)
        mu_s, sd_s = s.mean.reshape(shape), s.std.reshape(shape)
        mu_t, sd_t = t.mean.reshape(shape), t.std.reshape(shape)
        return like(arr, (x - mu_s) / sd_s * sd_t + mu_t)

    def _slot(self, slot: MemorySlot) -> MemorySlot:
        return slot.replace_tensors(
            maskmem=self._match(slot.maskmem, group_of(slot.kind, "maskmem")),
            obj_ptr=self._match(slot.obj_ptr, group_of(slot.kind, "obj_ptr")),
        )

    def translate(self, state: HandoffState) -> HandoffState:
        return state.clone().map_slots(self._slot, translator=self.name)
