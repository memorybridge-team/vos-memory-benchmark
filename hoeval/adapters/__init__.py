"""Track C's plug-in point, and the baselines that occupy it until track C
ships.

Everything here satisfies `interfaces.Translator`: state in, state out, no
knowledge of SAM 2 and no knowledge of the harness.  `Ours` lands in this
directory beside `NormMatched` and is scored by the same runner path, so there
is no route by which the proposed method could be measured more favourably than
the baselines it is compared against.
"""

from .identity import DirectCopy  # noqa: F401
from .norm_matched import NormMatched  # noqa: F401
from .stats import ChannelStats, GROUPS, StatsBook, group_of  # noqa: F401
