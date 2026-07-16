"""Market data for the XVA engine: synthetic defaults or real quotes.

Real mode pulls free, keyless sources:

  - USD discount curve: US Treasury par yields via FRED's CSV endpoint
    (used directly as zero rates - a demo simplification)
  - Henry Hub forward curve: NYMEX NG futures strip via Yahoo Finance
  - Commodity vol: realized vol of front-month NG futures (1y of daily
    log returns, annualized)

Fetched data is cached as JSON under ``data/processed`` (gitignored) so a
given as-of date hits the network once.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date

FRED_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DGS3MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10,DGS20,DGS30"
)
FRED_TENORS = (0.25, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0)
NG_MONTH_CODES = "FGHJKMNQUVXZ"  # Jan..Dec futures month codes


@dataclass
class MarketData:
    asof: date
    zeros: list[tuple[float, float]]     # (tenor years, zero rate)
    spot: float                          # commodity spot / front price
    forwards: list[tuple[date, float]]   # (delivery date, price), ascending
    sigma: float                         # lognormal commodity vol
    source: str = "synthetic"
    notes: list[str] = field(default_factory=list)

    def forward_at(self, d: date) -> float:
        """Linear interpolation on the forward curve, flat beyond the ends."""
        pts = self.forwards
        if d <= pts[0][0]:
            return pts[0][1]
        for (d0, p0), (d1, p1) in zip(pts, pts[1:]):
            if d <= d1:
                w = (d - d0).days / (d1 - d0).days
                return p0 + w * (p1 - p0)
        return pts[-1][1]

    def average_forward(self, start: date, end: date) -> float:
        """Mean forward over delivery points in [start, end] (swap fair price)."""
        window = [p for d, p in self.forwards if start <= d <= end]
        if not window:
            return self.forward_at(end)
        return sum(window) / len(window)

    def to_json(self) -> str:
        return json.dumps(
            {
                "asof": self.asof.isoformat(),
                "zeros": self.zeros,
                "spot": self.spot,
                "forwards": [(d.isoformat(), p) for d, p in self.forwards],
                "sigma": self.sigma,
                "source": self.source,
                "notes": self.notes,
            },
            indent=1,
        )

    @classmethod
    def from_json(cls, text: str) -> "MarketData":
        raw = json.loads(text)
        return cls(
            asof=date.fromisoformat(raw["asof"]),
            zeros=[tuple(z) for z in raw["zeros"]],
            spot=raw["spot"],
            forwards=[(date.fromisoformat(d), p) for d, p in raw["forwards"]],
            sigma=raw["sigma"],
            source=raw["source"],
            notes=raw.get("notes", []),
        )


def _add_months(d: date, months: int) -> date:
    y, m = divmod(d.month - 1 + months, 12)
    y += d.year
    m += 1
    leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(y, m, min(d.day, days[m - 1]))


def synthetic_market(
    asof: date,
    *,
    spot: float = 3.50,
    contango: float = 0.02,
    rate: float = 0.04,
    sigma: float = 0.35,
    horizon_years: float = 6.0,
) -> MarketData:
    """Flat rates + smooth contango curve (the original pedagogical setup)."""
    forwards = []
    for i in range(1, int(horizon_years * 4) + 1):
        d = _add_months(asof, 3 * i)
        t = (d - asof).days / 365.0
        forwards.append((d, spot * math.exp(contango * t)))
    return MarketData(
        asof=asof,
        zeros=[(1.0, rate)],
        spot=spot,
        forwards=forwards,
        sigma=sigma,
        source="synthetic",
    )


def _parse_fred_csv(text: str) -> tuple[date, list[tuple[float, float]]]:
    """Latest complete row of the FRED multi-series CSV → (date, zero curve)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines[1:]):
        cells = ln.split(",")
        if len(cells) == len(FRED_TENORS) + 1 and "." not in [c or "." for c in cells[1:]]:
            try:
                rates = [float(c) / 100.0 for c in cells[1:]]
            except ValueError:
                continue
            return date.fromisoformat(cells[0]), list(zip(FRED_TENORS, rates))
    raise ValueError("no complete row in FRED response")


def _ng_tickers(asof: date, months: int) -> list[tuple[date, str]]:
    """(delivery month, Yahoo ticker) for the NG strip after ``asof``."""
    out = []
    for i in range(1, months + 1):
        d = _add_months(date(asof.year, asof.month, 1), i)
        code = NG_MONTH_CODES[d.month - 1]
        out.append((d, f"NG{code}{d.year % 100:02d}.NYM"))
    return out


def _cache_path(asof: date, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"xva_market_{asof.strftime('%Y%m%d')}.json")


def fetch_real_market(
    asof: date | None = None,
    *,
    months: int = 36,
    cache_dir: str = "data/processed",
) -> MarketData:
    """Fetch treasuries (FRED), NG strip and realized vol (Yahoo), with cache."""
    asof = asof or date.today()
    cache_path = _cache_path(asof, cache_dir)
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return MarketData.from_json(f.read())

    # deferred: yfinance drags in curl_cffi, which hosts like Streamlit
    # must never load in-process (see fetch_real_market_isolated)
    import requests
    import yfinance as yf

    curve_date, zeros = _parse_fred_csv(requests.get(FRED_URL, timeout=30).text)

    strip = _ng_tickers(asof, months)
    tickers = ["NG=F"] + [t for _, t in strip]
    px = yf.download(tickers=tickers, period="5d", progress=False, auto_adjust=True)[
        "Close"
    ]
    last = px.ffill().iloc[-1]
    spot = float(last["NG=F"])
    forwards = [
        (d, float(last[t]))
        for d, t in strip
        if t in last.index and not math.isnan(float(last[t]))
    ]
    if len(forwards) < 4:
        raise RuntimeError(f"only {len(forwards)} NG contracts resolved on Yahoo")

    hist = yf.download(tickers="NG=F", period="1y", progress=False, auto_adjust=True)[
        "Close"
    ].dropna()
    rets = [math.log(b / a) for a, b in zip(hist.values.flat, hist.values.flat[1:])]
    mean = sum(rets) / len(rets)
    sigma = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)) * math.sqrt(252)

    md = MarketData(
        asof=asof,
        zeros=zeros,
        spot=spot,
        forwards=forwards,
        sigma=round(sigma, 4),
        source="FRED treasuries + Yahoo NYMEX NG",
        notes=[
            f"treasury curve date {curve_date.isoformat()} (par yields used as zeros)",
            f"{len(forwards)} NG contracts, realized vol from 1y front-month history",
        ],
    )
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        f.write(md.to_json())
    return md


def fetch_real_market_isolated(
    asof: date | None = None,
    *,
    months: int = 36,
    cache_dir: str = "data/processed",
) -> MarketData:
    """Fetch via a subprocess so network libs never load in this process.

    yfinance's curl_cffi (and other native deps) can corrupt allocator
    thread-state in multi-threaded hosts — a Streamlit rerun after loading
    it segfaulted inside pyarrow. The child fills the JSON cache; the
    parent only reads it back with stdlib json.
    """
    import subprocess
    import sys

    asof = asof or date.today()
    cache_path = _cache_path(asof, cache_dir)
    if not os.path.exists(cache_path):
        code = (
            "from datetime import date\n"
            "from creditlab.xva.marketdata import fetch_real_market\n"
            f"fetch_real_market(date.fromisoformat({asof.isoformat()!r}), "
            f"months={months}, cache_dir={cache_dir!r})\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        if not os.path.exists(cache_path):
            raise RuntimeError(f"market data fetch failed: {proc.stderr.strip()[-400:]}")
    with open(cache_path) as f:
        return MarketData.from_json(f.read())
