"""Model wrappers and the portable state they exchange.

`state` is import-safe anywhere.  `sam2_wrapper` imports torch and sam2, so it
is deliberately *not* imported here -- track B's tests and the translators must
keep running on a laptop with neither installed.
"""

from .state import HandoffState, MemorySlot, SLOT_PROMPTED, SLOT_RECENT, SLOT_KINDS  # noqa: F401
