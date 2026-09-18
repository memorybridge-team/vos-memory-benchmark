"""Transfer with no transformation at all.

`direct_copy` is the experiment the whole study hinges on, and it is free to
run.  SAM 2 memory tensors have the same shape at every model size (64
channels), so the small model's state can be installed in the large model
verbatim.  If that lands near Warm Target, there is no representation gap inside
the SAM 2 family and a translator has nothing to translate -- the study has to
move to a heterogeneous pair immediately.  Run it first.

The CMMT pilot scored 0 GT-visible J&F here across 10 switches on 3 videos.
Three videos is not DAVIS, which is exactly why it is being re-run.
"""

from __future__ import annotations

from ..models.state import HandoffState

__all__ = ["DirectCopy"]


class DirectCopy:
    """Satisfies `interfaces.Translator` and changes nothing."""

    def __init__(self, target_model: str, source_model: str = "*"):
        self.source_model = source_model
        self.target_model = target_model
        self.name = "direct_copy"

    def translate(self, state: HandoffState) -> HandoffState:
        return state.clone()
