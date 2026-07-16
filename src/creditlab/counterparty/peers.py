"""Energy-sector peer sets: percentile context for counterparty ratios.

Groups the scored universe into desk-style energy peer sets by SIC code and
ranks a counterparty's key ratios against its peers. Trading credit memos cite
this ("leverage 80th percentile of upstream peers") to anchor a name's
fundamentals in sector context rather than absolute thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered: first matching SIC prefix wins (492x is midstream, not generic 49xx).
PEER_SETS: dict[str, tuple[str, ...]] = {
    "upstream_oil_gas": ("131", "132"),
    "oilfield_services": ("138",),
    "refining_marketing": ("29", "517"),
    "midstream_pipelines": ("46", "492"),
    "power_utilities": ("49",),
}
PEER_SET_LABELS = {
    "upstream_oil_gas": "Upstream oil & gas (E&P)",
    "oilfield_services": "Oilfield services & drilling",
    "refining_marketing": "Refining & petroleum marketing",
    "midstream_pipelines": "Midstream & pipelines",
    "power_utilities": "Power & utilities",
}
# (ratio column, reading direction for credit quality)
PEER_RATIOS = [
    ("leverage", "higher=weaker"),
    ("interest_coverage", "higher=stronger"),
    ("current_ratio", "higher=stronger"),
    ("roa", "higher=stronger"),
    ("cfo_to_debt", "higher=stronger"),
]


def peer_set_for(sic: float | str | None) -> str | None:
    """Map a 4-digit SIC code to its energy peer set (None if non-energy)."""
    if sic is None or (isinstance(sic, float) and np.isnan(sic)):
        return None
    code = str(int(float(sic)))
    for name, prefixes in PEER_SETS.items():
        if code.startswith(prefixes):
            return name
    return None


def peer_universe(universe: pd.DataFrame, set_name: str) -> pd.DataFrame:
    """All rows of ``universe`` belonging to the given peer set."""
    mask = universe["sic"].map(lambda s: peer_set_for(s) == set_name)
    return universe[mask]


def peer_percentiles(row: pd.Series, universe: pd.DataFrame) -> dict | None:
    """Percentile rank of the counterparty's ratios within its energy peer set.

    Returns None when the counterparty's SIC has no energy peer set. The row
    itself is excluded from the peer pool when present (matched on cik).
    """
    set_name = peer_set_for(row.get("sic"))
    if set_name is None:
        return None
    peers = peer_universe(universe, set_name)
    if "cik" in peers.columns and not pd.isna(row.get("cik")):
        peers = peers[peers["cik"] != row["cik"]]
    percentiles: dict[str, float] = {}
    for ratio, _ in PEER_RATIOS:
        x = float(row.get(ratio, np.nan))
        vals = peers[ratio].dropna() if ratio in peers.columns else pd.Series(dtype=float)
        if np.isnan(x) or vals.empty:
            continue
        percentiles[ratio] = float((vals <= x).mean())
    return {
        "peer_set": set_name,
        "label": PEER_SET_LABELS[set_name],
        "n_peers": int(len(peers)),
        "percentiles": percentiles,
    }


def peer_context_lines(row: pd.Series, universe: pd.DataFrame) -> list[str]:
    """Human-readable peer-context lines for a credit memo / CLI printout."""
    ctx = peer_percentiles(row, universe)
    if ctx is None:
        sic = row.get("sic")
        return [f"No energy peer set for SIC {sic} — peer context skipped."]
    lines = [f"{ctx['label']} peers (n={ctx['n_peers']}):"]
    directions = dict(PEER_RATIOS)
    for ratio, pct in ctx["percentiles"].items():
        lines.append(f"  {ratio}: {pct:.0%} percentile ({directions[ratio]})")
    return lines
