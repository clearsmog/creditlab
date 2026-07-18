"""Loader for LSEG Workspace / Codebook exports.

Codebook (the Jupyter environment inside LSEG Workspace) runs
``notebooks/codebook_cds_pull.ipynb`` to pull a USD SOFR zero curve, the
Henry Hub futures strip, an ATM implied vol, and per-ticker CDS spread
curves, then writes one JSON export. This module reads that file locally —
no LSEG libraries or entitlements needed on this side.

Keep exports under ``data/processed/`` (gitignored): the academic licence
permits personal research use, not redistribution.

Export schema (version 1):

  {
    "version": 1,
    "asof": "2026-07-17",
    "source": "LSEG Workspace Codebook",
    "zeros": [[0.25, 0.0381], [1.0, 0.0398], ...],
    "gas": {"spot": 2.85, "forwards": [["2026-08-01", 2.81], ...],
            "implied_vol": 0.62},
    "cds": {"OXY": {"recovery": 0.4,
                    "spreads": [[1.0, 0.005], [5.0, 0.009], ...]}},
    "notes": ["free text provenance"]
  }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from creditlab.xva.marketdata import MarketData


@dataclass
class CdsQuote:
    """One counterparty's CDS term structure (tenor years, spread decimal)."""

    spreads: list[tuple[float, float]]
    recovery: float = 0.4


@dataclass
class LsegExport:
    asof: date
    market: MarketData
    cds: dict[str, CdsQuote] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def cds_for(self, ticker: str) -> CdsQuote | None:
        return self.cds.get(ticker.upper())


def load_lseg_export(path: str) -> LsegExport:
    with open(path) as f:
        raw = json.load(f)
    if raw.get("version") != 1:
        raise ValueError(f"unsupported LSEG export version: {raw.get('version')!r}")

    asof = date.fromisoformat(raw["asof"])
    gas = raw["gas"]
    market = MarketData(
        asof=asof,
        zeros=[tuple(z) for z in raw["zeros"]],
        spot=float(gas["spot"]),
        forwards=[(date.fromisoformat(d), float(p)) for d, p in gas["forwards"]],
        sigma=float(gas["implied_vol"]),
        source=raw.get("source", "LSEG Workspace export"),
        notes=list(raw.get("notes", [])),
    )
    cds = {
        tk.upper(): CdsQuote(
            spreads=[tuple(s) for s in q["spreads"]],
            recovery=float(q.get("recovery", 0.4)),
        )
        for tk, q in raw.get("cds", {}).items()
        if q.get("spreads")
    }
    return LsegExport(asof=asof, market=market, cds=cds, notes=market.notes)
