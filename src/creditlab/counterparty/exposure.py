"""Simple pre-deal exposure helpers for trading-credit demos.

These are *illustrative* add-on style measures, not full Monte Carlo PFE.
Useful for FO dialogue: 'does this ticket fit inside the proposed limit?'
"""

from __future__ import annotations

import math


def pfe_addon(
    notional: float,
    tenor_years: float,
    *,
    annual_vol: float = 0.35,
    conf_z: float = 1.65,
) -> float:
    """Rough PE/PFE-style add-on: notional × σ × √T × z (one-factor lognormal proxy).

    Default vol 35% is a ballpark energy-commodity number for demos only.
    """
    if notional <= 0 or tenor_years <= 0:
        return 0.0
    return float(notional * annual_vol * math.sqrt(tenor_years) * conf_z)


def headroom(limit_usd: float, current_exposure_usd: float, pfe_usd: float) -> dict:
    """Limit utilisation after booking a deal with given PFE add-on."""
    after = current_exposure_usd + pfe_usd
    free = limit_usd - after
    util = after / limit_usd if limit_usd > 0 else float("inf")
    return {
        "limit_usd": limit_usd,
        "current_exposure_usd": current_exposure_usd,
        "pfe_usd": pfe_usd,
        "post_deal_exposure_usd": after,
        "headroom_usd": free,
        "utilisation": util,
        "breach": free < 0 or limit_usd <= 0,
    }
