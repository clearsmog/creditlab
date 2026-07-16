"""Trading-credit desk tools: counterparty assessment, limit policy, FO memo.

Framed for energy-merchant / commodity-trading credit workflows (limits,
documentation, pre-deal exposure vs limit) rather than bank IRB capital.
"""

from creditlab.counterparty.blotter import build_limit_blotter
from creditlab.counterparty.limits import (
    LimitRecommendation,
    assess_ratios,
    recommend_limit,
)
from creditlab.counterparty.memo import format_credit_memo
from creditlab.counterparty.exposure import pfe_addon, headroom
from creditlab.counterparty.peers import (
    peer_context_lines,
    peer_percentiles,
    peer_set_for,
)

__all__ = [
    "LimitRecommendation",
    "assess_ratios",
    "build_limit_blotter",
    "recommend_limit",
    "format_credit_memo",
    "pfe_addon",
    "headroom",
    "peer_context_lines",
    "peer_percentiles",
    "peer_set_for",
]
