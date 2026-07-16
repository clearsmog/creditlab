"""CVA/PFE via ORE (Open Source Risk Engine): true counterparty risk.

Simulates a synthetic gas netting set (forward + fixed-price swap) under an
LGM x Schwartz cross-asset model and post-processes EPE/PFE profiles and CVA,
with the counterparty default curve implied from the CreditLab model PD.

ORE is an optional dependency: ``uv sync --extra xva``. Importing this
package is safe without it; only ``run_xva`` requires the bindings.
"""

from creditlab.xva.configs import XvaInputs
from creditlab.xva.marketdata import MarketData, fetch_real_market, synthetic_market
from creditlab.xva.runner import XvaResults, run_xva

__all__ = [
    "MarketData",
    "XvaInputs",
    "XvaResults",
    "fetch_real_market",
    "run_xva",
    "synthetic_market",
]
