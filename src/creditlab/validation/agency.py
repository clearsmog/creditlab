"""Benchmark the internal scorecard rating against S&P issuer ratings.

Input is a Capital IQ screening export (CSV) with a ticker column and an
S&P long-term issuer credit rating column — see the README for the exact
screening recipe. Keep exports in ``data/processed/`` (gitignored):
academic Capital IQ access does not permit redistribution.

Agency notched ratings (AA-, BBB+, …) are collapsed onto the internal
7-grade scale before comparison; D/SD (defaulted) and NR are excluded from
the statistics but counted.

CLI:

  uv run python -m creditlab.validation.agency data/processed/capiq_ratings.csv
  uv run python -m creditlab.validation.agency --dump-tickers tickers.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from creditlab.portfolio.ratings import GRADES

GRADE_ORD = {g: i for i, g in enumerate(GRADES)}

AGENCY_TO_GRADE: dict[str, str | None] = {
    "AAA": "AAA",
    "AA+": "AA", "AA": "AA", "AA-": "AA",
    "A+": "A", "A": "A", "A-": "A",
    "BBB+": "BBB", "BBB": "BBB", "BBB-": "BBB",
    "BB+": "BB", "BB": "BB", "BB-": "BB",
    "B+": "B", "B": "B", "B-": "B",
    "CCC+": "CCC", "CCC": "CCC", "CCC-": "CCC", "CC": "CCC", "C": "CCC",
    "SD": None, "D": None, "NR": None,  # excluded from comparison
}


def _find_header_row(path: str) -> int:
    """CapIQ exports often carry preamble rows before the real header."""
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            cells = [c.strip().lower() for c in line.split(",")]
            if any("ticker" in c for c in cells):
                return i
            if i > 20:
                break
    raise ValueError(f"no header row with a ticker column found in {path}")


def load_agency_ratings(path: str) -> pd.DataFrame:
    """Read a CapIQ export → DataFrame[ticker, agency_rating, agency_grade]."""
    df = pd.read_csv(path, skiprows=_find_header_row(path), encoding="utf-8-sig")
    cols = {c.lower().strip(): c for c in df.columns}
    ticker_col = next(c for k, c in cols.items() if "ticker" in k)
    rating_col = next(c for k, c in cols.items() if "rating" in k)

    out = pd.DataFrame({
        "ticker": df[ticker_col].astype(str).str.strip().str.upper(),
        # CapIQ tickers come as "NYSE:OXY" — keep the symbol part
        "agency_rating": df[rating_col].astype(str).str.strip().str.upper(),
    })
    out["ticker"] = out["ticker"].str.split(":").str[-1]
    out = out[(out["ticker"] != "") & (out["ticker"] != "NAN")]
    out["agency_grade"] = out["agency_rating"].map(AGENCY_TO_GRADE)
    return out.drop_duplicates(subset="ticker").reset_index(drop=True)


@dataclass
class AgencyComparison:
    n_agency: int              # rows in the export
    n_matched: int             # matched to the scored panel, with usable rating
    n_excluded: int            # matched but NR / D / SD
    exact: float               # share with identical coarse grade
    within_one: float          # share within one grade
    spearman: float
    kendall: float
    mean_signed_diff: float    # model_ord - agency_ord; >0 → model harsher
    confusion: pd.DataFrame    # rows = model grade, cols = agency grade
    detail: pd.DataFrame       # per-name: ticker, model, agency, diff


def compare_ratings(scored: pd.DataFrame, agency: pd.DataFrame) -> AgencyComparison:
    """Compare internal ratings (``rating`` column) to agency coarse grades.

    ``scored`` is one row per issuer with ``ticker`` and ``rating``
    (e.g. ``load_scored_latest()``).
    """
    left = scored[["ticker", "rating"]].copy()
    left["ticker"] = left["ticker"].astype(str).str.upper()
    m = left.merge(agency, on="ticker", how="inner")
    n_excluded = int(m["agency_grade"].isna().sum())
    m = m.dropna(subset=["agency_grade"])

    model_ord = m["rating"].map(GRADE_ORD)
    agency_ord = m["agency_grade"].map(GRADE_ORD)
    diff = model_ord - agency_ord

    if len(m) >= 3:
        sp, _ = spearmanr(model_ord, agency_ord)
        kt, _ = kendalltau(model_ord, agency_ord)
    else:
        sp = kt = float("nan")

    confusion = (
        pd.crosstab(m["rating"], m["agency_grade"])
        .reindex(index=GRADES, columns=GRADES, fill_value=0)
    )
    detail = m.assign(notch_diff=diff).sort_values("notch_diff")

    return AgencyComparison(
        n_agency=len(agency),
        n_matched=len(m),
        n_excluded=n_excluded,
        exact=float((diff == 0).mean()),
        within_one=float((diff.abs() <= 1).mean()),
        spearman=float(sp),
        kendall=float(kt),
        mean_signed_diff=float(diff.mean()),
        confusion=confusion,
        detail=detail.reset_index(drop=True),
    )


def format_report(c: AgencyComparison) -> str:
    header = [
        "# Scorecard vs S&P issuer ratings",
        f"Agency export: {c.n_agency} names | matched to panel: {c.n_matched} "
        f"(+{c.n_excluded} matched but NR/D — excluded)",
    ]
    if c.n_matched == 0:
        return "\n".join(header + [
            "",
            "No overlap between the export and the panel's tickers — check that "
            "the export was screened on the list from --dump-tickers.",
        ])
    bias = "harsher" if c.mean_signed_diff > 0 else "more lenient"
    lines = header + [
        "",
        f"Exact grade match: {c.exact:.0%} | within one grade: {c.within_one:.0%}",
        f"Rank correlation: Spearman {c.spearman:.2f}, Kendall {c.kendall:.2f}",
        f"Mean signed diff: {c.mean_signed_diff:+.2f} grades "
        f"(model {bias} than S&P on average)",
        "",
        "Confusion matrix (rows = model, cols = S&P):",
        c.confusion.to_string(),
    ]
    worst = c.detail[c.detail["notch_diff"].abs() >= 2]
    if len(worst):
        lines += ["", f"Names ≥2 grades apart ({len(worst)}):"]
        for _, r in worst.iterrows():
            lines.append(
                f"  {r['ticker']:<6} model {r['rating']:<4} vs S&P "
                f"{r['agency_grade']:<4} ({r['agency_rating']})"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="data/processed/capiq_ratings.csv",
                        help="Capital IQ screening export")
    parser.add_argument("--dump-tickers", metavar="OUT",
                        help="write the panel ticker list (for CapIQ upload) and exit")
    args = parser.parse_args()

    from creditlab.counterparty.desk import load_scored_latest

    scored = load_scored_latest()
    if args.dump_tickers:
        tickers = sorted(t for t in scored["ticker"].dropna().astype(str) if t)
        pd.DataFrame({"ticker": tickers}).to_csv(args.dump_tickers, index=False)
        print(f"wrote {len(tickers)} tickers → {args.dump_tickers}")
        return

    agency = load_agency_ratings(args.csv)
    print(format_report(compare_ratings(scored, agency)))


if __name__ == "__main__":
    main()
