"""FAME private-counterparty book: UK unlisted names through the desk pipeline.

FAME (Bureau van Dijk) covers UK & Irish private companies — the population an
energy desk actually faces. A FAME screening export (energy SIC codes,
financial columns, FAME's own credit score) is mapped into the panel ratio
schema and pushed through the real pipeline: scorecard PD → master-scale
rating → unsecured limit policy. FAME's score rides along as an external
sanity check.

Population-transfer caveat, printed on every run: the scorecard was developed
on US listed issuers; applying it to UK private companies is a demo of the
mechanics, not a validated model for that population.

CLI:

  uv run python -m creditlab.counterparty.fame data/processed/fame_export.xlsx
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from creditlab.counterparty.blotter import build_limit_blotter
from creditlab.models.scorecard import (
    CENTRAL_TENDENCY,
    RATIO_FEATURES,
    Scorecard,
    calibrate_pds,
)
from creditlab.portfolio.ratings import assign_rating

# header fragment (lowercase) → output column
FAME_COLUMNS = {
    "company name": "name",
    "registered number": "reg_number",
    "sic": "sic_text",
    "total assets": "assets_gbp_th",
    "shareholders funds": "equity_gbp_th",
    "turnover": "turnover_gbp_th",
    "current ratio": "current_ratio",
    "interest cover": "interest_coverage",
    "return on total assets": "roa_pct",
    "gearing": "gearing_pct",
    "solvency": "solvency_pct",
    "score": "fame_score",
    "latest accounts date": "accounts_date",
}


def _read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        # calamine, not openpyxl: FAME styles.xml uses attributes openpyxl
        # rejects. Real exports ship a "Search summary" sheet before the
        # results, so scan sheets for the one with a company-name header.
        book = pd.read_excel(path, header=None, sheet_name=None, engine="calamine")
        for raw in book.values():
            hdr = next(
                (i for i in range(min(len(raw), 25))
                 if raw.iloc[i].astype(str).str.contains("company name", case=False).any()),
                None,
            )
            if hdr is not None:
                df = raw.iloc[hdr + 1:].reset_index(drop=True)
                df.columns = [str(c) for c in raw.iloc[hdr]]
                return df
        raise ValueError(f"no sheet with a company-name header in {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False), errors="coerce"
    )


def load_fame_export(path: str, fx_gbpusd: float = 1.27) -> pd.DataFrame:
    """FAME export (xlsx/csv) → one row per company in the panel ratio schema.

    FAME reports financials in thousands of GBP and ratios as percentages;
    values are converted to USD and decimals to match the EDGAR panel.
    "n.a." cells become NaN — the scorecard's WoE binning has a missing bin.
    """
    raw = _read_table(path)
    cols: dict[str, str] = {}
    for col in raw.columns:
        low = str(col).replace("\n", " ").lower()
        for frag, out_name in FAME_COLUMNS.items():
            if frag in low and out_name not in cols:
                cols[out_name] = col
    if "name" not in cols:
        raise ValueError(f"no company-name column found in {path}")

    out = pd.DataFrame({"name": raw[cols["name"]].astype(str).str.strip()})
    out = out[(out["name"] != "") & (out["name"].str.lower() != "nan")]
    for key in ("reg_number", "sic_text", "accounts_date"):
        if key in cols:
            out[key] = raw.loc[out.index, cols[key]]
    for key in ("assets_gbp_th", "equity_gbp_th", "turnover_gbp_th", "current_ratio",
                "interest_coverage", "roa_pct", "gearing_pct", "solvency_pct",
                "fame_score"):
        if key in cols:
            out[key] = _num(raw.loc[out.index, cols[key]])

    # panel schema: USD levels, decimal ratios
    out["assets"] = out.get("assets_gbp_th", np.nan) * 1_000 * fx_gbpusd
    out["equity"] = out.get("equity_gbp_th", np.nan) * 1_000 * fx_gbpusd
    out["roa"] = out.get("roa_pct", np.nan) / 100.0
    out["debt_to_equity"] = out.get("gearing_pct", np.nan) / 100.0
    solvency = out.get("solvency_pct", np.nan) / 100.0
    out["leverage"] = 1.0 - solvency
    fallback = 1.0 - out["equity"] / out["assets"]
    out["leverage"] = out["leverage"].fillna(fallback)
    out["asset_turnover"] = (
        out.get("turnover_gbp_th", np.nan) * 1_000 * fx_gbpusd / out["assets"]
    )
    out["log_assets"] = np.log(out["assets"].where(out["assets"] > 0))
    if "accounts_date" in out:
        out["period_end"] = pd.to_datetime(
            out["accounts_date"], errors="coerce", dayfirst=True  # UK dd/mm/yyyy
        )
    return out.reset_index(drop=True)


def score_private_book(
    book: pd.DataFrame, panel_path: str = "data/processed/panel.parquet"
) -> pd.DataFrame:
    """Score FAME rows with the panel-trained scorecard → pd_cal + rating."""
    df = pd.read_parquet(panel_path)
    train = df[df["fyear"] <= 2019]
    card = Scorecard().fit(train, train["default_within_1y"])
    sample_rate = float(train["default_within_1y"].mean())

    book = book.copy()
    for f in RATIO_FEATURES:
        if f not in book:
            book[f] = np.nan
    pd_cal = calibrate_pds(card.predict_pd(book), sample_rate, CENTRAL_TENDENCY)
    book["pd_cal"] = pd_cal
    book["rating"] = assign_rating(pd_cal)
    return book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", nargs="?", default="data/processed/fame_export.xlsx",
                        help="FAME screening export (xlsx or csv)")
    parser.add_argument("--fx", type=float, default=1.27, help="GBP/USD rate")
    args = parser.parse_args()

    book = load_fame_export(args.export, fx_gbpusd=args.fx)
    scored = score_private_book(book)
    blotter = build_limit_blotter(scored)

    print("# FAME private-counterparty book")
    print("CAVEAT: scorecard developed on US listed issuers — applying it to UK "
          "private names\ndemonstrates the pipeline, it is not a validated "
          "cross-population model.\n")
    show = blotter.merge(
        scored[["name", "fame_score"]].drop_duplicates("name"), on="name", how="left"
    ) if "fame_score" in scored else blotter.assign(fame_score=np.nan)
    cols = ["name", "rating", "pd_1y", "fame_score", "recommended_limit_usd",
            "max_tenor_years", "kyc_status"]
    with pd.option_context("display.width", 140, "display.max_colwidth", 32):
        print(show[cols].to_string(index=False,
                                   formatters={"pd_1y": "{:.2%}".format,
                                               "recommended_limit_usd": "{:,.0f}".format}))

    ok = show.dropna(subset=["fame_score"])
    if len(ok) >= 5:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(ok["pd_1y"], ok["fame_score"])
        direction = "consistent" if rho < 0 else "OPPOSED — check score orientation"
        print(f"\nSpearman(model PD, FAME score) = {rho:+.2f} on {len(ok)} names "
              f"(FAME scores are higher=safer, so negative = {direction})")
    total = float(show["recommended_limit_usd"].sum())
    n_lines = int((show["recommended_limit_usd"] > 0).sum())
    print(f"\n{len(show)} names | proposed unsecured capacity ${total/1e6:,.1f}m "
          f"| {n_lines} names with a line")


if __name__ == "__main__":
    main()
