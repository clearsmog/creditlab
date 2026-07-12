"""Merton structural model: distance-to-default from equity market data.

Equity is a call option on the firm's assets struck at the debt barrier:

    E = V N(d1) - F e^{-rT} N(d2)
    sigma_E = (V / E) N(d1) sigma_V          (Ito, applied to E(V))

Observables are market cap E, equity vol sigma_E, and the default point F
(KMV convention: current liabilities + half of long-term debt). The two
equations are solved for the unobservables (V, sigma_V); then

    DD = (ln(V/F) + (mu - sigma_V^2 / 2) T) / (sigma_V sqrt(T))
    PD = N(-DD)   (risk-neutral when mu = r)

This is the market-implied counterpart to the fundamentals scorecard: it
reacts daily, while ratios update annually.

Demo (COVID stress on real firms):  uv run python -m creditlab.models.merton
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from scipy.stats import norm


def merton_system(v: float, sigma_v: float, e: float, sigma_e: float,
                  f: float, r: float, t: float = 1.0) -> tuple[float, float]:
    """Residuals of the two Merton equations at (V, sigma_V)."""
    d1 = (np.log(v / f) + (r + sigma_v**2 / 2) * t) / (sigma_v * np.sqrt(t))
    d2 = d1 - sigma_v * np.sqrt(t)
    eq_value = v * norm.cdf(d1) - f * np.exp(-r * t) * norm.cdf(d2) - e
    eq_vol = (v / e) * norm.cdf(d1) * sigma_v - sigma_e
    return eq_value, eq_vol


def solve_merton(e: float, sigma_e: float, f: float, r: float = 0.03,
                 t: float = 1.0) -> tuple[float, float, float, float]:
    """Solve for (V, sigma_V); return (V, sigma_V, DD, PD).

    Solved in log-asset space for numerical stability. Initial guess:
    V ~ E + F (balance-sheet identity), sigma_V ~ sigma_E * E / (E + F)
    (delta ~ 1 deleveraging of equity vol).
    """
    def residuals(params):
        log_v, sigma_v = params
        return merton_system(np.exp(log_v), abs(sigma_v), e, sigma_e, f, r, t)

    x0 = (np.log(e + f), sigma_e * e / (e + f))
    (log_v, sigma_v), info, ok, _ = fsolve(residuals, x0, full_output=True)
    if ok != 1:
        return np.nan, np.nan, np.nan, np.nan
    v, sigma_v = float(np.exp(log_v)), float(abs(sigma_v))
    dd = (np.log(v / f) + (r - sigma_v**2 / 2) * t) / (sigma_v * np.sqrt(t))
    return v, sigma_v, float(dd), float(norm.cdf(-dd))


def firm_dd_series(prices: pd.Series, shares: float, f: float,
                   r: float = 0.03, vol_window: int = 252) -> pd.DataFrame:
    """Month-end DD/PD series for one firm from a daily price series."""
    returns = np.log(prices / prices.shift(1))
    sigma_e = returns.rolling(vol_window).std() * np.sqrt(252)
    market_cap = prices * shares

    rows = []
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index
    for dt in month_ends:
        e, s = float(market_cap.loc[dt]), float(sigma_e.loc[dt])
        if np.isnan(s) or e <= 0 or f <= 0:
            continue
        v, sigma_v, dd, pd_ = solve_merton(e, s, f, r)
        rows.append({"date": dt, "equity": e, "sigma_e": s,
                     "asset_value": v, "sigma_v": sigma_v, "dd": dd, "pd": pd_})
    return pd.DataFrame(rows).set_index("date")


def default_point(row: pd.Series) -> float:
    """KMV default point: current liabilities + 0.5 * long-term debt."""
    cl = row.get("current_liabilities") or 0.0
    ltd = row.get("long_term_debt") or 0.0
    if not cl and not ltd:  # unclassified balance sheet: fall back to total debt
        return float(row.get("liabilities") or 0.0)
    return float(cl + 0.5 * ltd)


def main() -> None:
    import yfinance as yf

    from creditlab.data.edgar import build_panel

    demo = ["AAPL", "F", "CCL", "AAL"]
    panel = build_panel(demo)  # served from the EDGAR cache

    print("Merton DD through the COVID shock (fiscal-2019 balance sheets):\n")
    print(f"{'ticker':8s} {'2019-12':>8s} {'2020-03':>8s} {'2020-12':>8s}   PD(2020-03)")
    for ticker in demo:
        firm = panel[(panel["ticker"] == ticker) & (panel["fyear"] == 2019)]
        if firm.empty:
            continue
        row = firm.iloc[0]
        f_point = default_point(row)
        shares = row["shares_outstanding"]
        prices = yf.download(ticker, start="2018-06-01", end="2021-01-15",
                             progress=False, auto_adjust=True)["Close"][ticker]
        dd = firm_dd_series(prices, shares, f_point)
        pick = lambda ym: dd[dd.index.to_period("M") == ym]
        vals = [pick(ym) for ym in ("2019-12", "2020-03", "2020-12")]
        cells = [f"{v['dd'].iloc[0]:8.2f}" if len(v) else "     n/a" for v in vals]
        pd_mar = f"{vals[1]['pd'].iloc[0]:.2%}" if len(vals[1]) else "n/a"
        print(f"{ticker:8s} {cells[0]} {cells[1]} {cells[2]}   {pd_mar}")


if __name__ == "__main__":
    main()
