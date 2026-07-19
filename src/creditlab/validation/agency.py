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
import re
from dataclasses import dataclass

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from creditlab.portfolio.ratings import GRADES

# legal-form tokens stripped (iteratively) from the end of normalized names
NAME_SUFFIXES = {"INC", "CORP", "CORPORATION", "INCORPORATED", "CO", "COMPANY",
                 "LLC", "LP", "LTD", "LIMITED", "PLC"}


def normalize_name(s: str) -> str:
    """Company name → canonical form for exact matching (no fuzzy logic:
    a false positive in a validation set is worse than a missed match)."""
    s = re.sub(r"\([^)]*\)", " ", str(s).upper())   # drop "(NYSE:XYZ)" etc.
    s = re.sub(r"[^A-Z0-9& ]", " ", s)
    toks = s.split()
    while len(toks) > 1 and toks[-1] in NAME_SUFFIXES:
        toks.pop()
    return " ".join(toks)

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


def _read_table(path: str) -> pd.DataFrame:
    """CSV or xlsx CapIQ export → raw DataFrame with real headers."""
    if path.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(path, header=None)
        hdr = next(
            i for i in range(min(len(raw), 25))
            if raw.iloc[i].astype(str).str.contains("ticker", case=False).any()
        )
        df = raw.iloc[hdr + 1:].reset_index(drop=True)
        df.columns = [str(c) for c in raw.iloc[hdr]]
        return df
    return pd.read_csv(path, skiprows=_find_header_row(path), encoding="utf-8-sig")


def load_agency_ratings(path: str) -> pd.DataFrame:
    """Read a CapIQ export → DataFrame[ticker, name, nname, agency_rating,
    agency_grade].

    Rows whose rating cell is not a recognised agency token (metadata rows,
    field aliases) are dropped. Tickers come from the Ticker column, falling
    back to a "(NYSE:XYZ)" parenthetical in the entity name. Normalized
    names that appear with conflicting ratings (parent vs subsidiary) are
    blanked so the name-matching stage cannot pick a wrong entity.
    """
    df = _read_table(path)
    cols = {c.lower().strip(): c for c in df.columns}
    ticker_col = next(c for k, c in cols.items() if "ticker" in k)
    rating_col = next(
        c for k, c in cols.items()
        if "rating" in k and "date" not in k and "action" not in k
    )
    name_col = next((c for k, c in cols.items() if "name" in k), None)

    out = pd.DataFrame({
        "ticker": df[ticker_col],
        "name": df[name_col] if name_col else "",
        "agency_rating": df[rating_col],
    }).dropna(subset=["agency_rating"])
    out["agency_rating"] = out["agency_rating"].astype(str).str.strip().str.upper()
    out = out[out["agency_rating"].isin(AGENCY_TO_GRADE)]
    out["agency_grade"] = out["agency_rating"].map(AGENCY_TO_GRADE)

    out["name"] = out["name"].fillna("").astype(str)
    # ticker: explicit column first ("NYSE:OXY" → OXY), else embedded in name
    out["ticker"] = (
        out["ticker"].astype(str).str.strip().str.upper().str.split(":").str[-1]
    )
    embedded = out["name"].str.upper().str.extract(r"\(\w+:([\w.]+)\)")[0]
    bad = out["ticker"].isin(["", "NAN", "NONE"])
    out.loc[bad, "ticker"] = embedded[bad]
    out["ticker"] = out["ticker"].fillna("")

    out["nname"] = out["name"].map(normalize_name)
    conflicted = (
        out[out["nname"] != ""]
        .groupby("nname")["agency_grade"]
        .nunique(dropna=False)  # NR/D vs a real grade is also a conflict
    )
    out.loc[out["nname"].isin(conflicted[conflicted > 1].index), "nname"] = ""

    out = out[(out["ticker"] != "") | (out["nname"] != "")]
    return out.reset_index(drop=True)


@dataclass
class AgencyComparison:
    n_agency: int              # usable rows in the export
    n_matched: int             # matched to the scored panel, with usable rating
    n_by_ticker: int
    n_by_name: int
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

    ``scored`` is one row per issuer with ``ticker``, ``rating`` and
    optionally ``name`` (e.g. ``load_scored_latest()``). Matching runs in two
    stages: exact ticker, then exact normalized company name.
    """
    left = scored.reset_index(drop=True).reset_index(names="_i")
    left["ticker"] = left["ticker"].fillna("").astype(str).str.upper()
    left["nname"] = (
        left["name"].fillna("").map(normalize_name) if "name" in left else ""
    )

    a_cols = ["agency_rating", "agency_grade"]
    by_ticker = agency[agency["ticker"] != ""].drop_duplicates("ticker")
    m1 = left[left["ticker"] != ""].merge(
        by_ticker[["ticker"] + a_cols], on="ticker", how="inner"
    )
    rest = left[~left["_i"].isin(m1["_i"])]
    by_name = agency[agency["nname"] != ""].drop_duplicates("nname")
    m2 = rest[rest["nname"] != ""].merge(
        by_name[["nname"] + a_cols], on="nname", how="inner"
    )
    m = pd.concat(
        [m1.assign(match="ticker"), m2.assign(match="name")], ignore_index=True
    )
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
        n_by_ticker=int((m["match"] == "ticker").sum()),
        n_by_name=int((m["match"] == "name").sum()),
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
        f"({c.n_by_ticker} by ticker, {c.n_by_name} by name; "
        f"+{c.n_excluded} matched but NR/D — excluded)",
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
            label = r["ticker"] or str(r.get("name", ""))[:20]
            lines.append(
                f"  {label:<20} model {r['rating']:<4} vs S&P "
                f"{r['agency_grade']:<4} ({r['agency_rating']})"
            )
    return "\n".join(lines)


def tune_central_tendency(agency: pd.DataFrame) -> None:
    """Sweep the calibration central tendency; report rating bias at each level.

    The remediation knob for a lenient/harsh scorecard: raising the target
    portfolio PD shifts every calibrated PD up, pushing ratings down the
    scale. Prints the bias curve and the level that zeroes the mean signed
    grade difference vs the agency benchmark.
    """
    from creditlab.models.scorecard import CENTRAL_TENDENCY, Scorecard, calibrate_pds
    from creditlab.portfolio.ratings import assign_rating

    df = pd.read_parquet("data/processed/panel.parquet")
    df = df[df["period_end"] <= pd.Timestamp.today() - pd.DateOffset(years=1)]
    train = df[df["fyear"] <= 2019]
    card = Scorecard().fit(train, train["default_within_1y"])
    sample_rate = float(train["default_within_1y"].mean())
    df = df.assign(pd_raw=card.predict_pd(df))
    latest = df.sort_values("period_end").groupby("cik", as_index=False).tail(1)
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=3)
    latest = latest[latest["period_end"] >= cutoff]

    print(f"{'CT':>6} {'bias':>7} {'exact':>6} {'within1':>8} {'matched':>8}")
    best = (None, float("inf"))
    for ct in (0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.060):
        scored = latest.assign(
            rating=assign_rating(
                calibrate_pds(latest["pd_raw"].to_numpy(), sample_rate, ct)
            )
        )
        c = compare_ratings(scored, agency)
        marker = " ← current" if abs(ct - CENTRAL_TENDENCY) < 1e-9 else ""
        print(f"{ct:>6.3f} {c.mean_signed_diff:>+7.2f} {c.exact:>6.0%} "
              f"{c.within_one:>8.0%} {c.n_matched:>8}{marker}")
        if abs(c.mean_signed_diff) < best[1]:
            best = (ct, abs(c.mean_signed_diff))
    print(f"\nbias-minimizing central tendency: {best[0]:.3f} "
          f"(set CENTRAL_TENDENCY in creditlab/models/scorecard.py)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="data/processed/capiq_ratings.csv",
                        help="Capital IQ screening export")
    parser.add_argument("--dump-tickers", metavar="OUT",
                        help="write the panel ticker list (for CapIQ upload) and exit")
    parser.add_argument("--tune-ct", action="store_true",
                        help="sweep the calibration central tendency against the benchmark")
    args = parser.parse_args()

    if args.tune_ct:
        tune_central_tendency(load_agency_ratings(args.csv))
        return

    from creditlab.counterparty.desk import load_scored_latest

    scored = load_scored_latest()
    if args.dump_tickers:
        tickers = sorted(t for t in scored["ticker"].dropna().astype(str) if t)
        pd.DataFrame({"ticker": tickers}).to_csv(args.dump_tickers, index=False)
        print(f"wrote {len(tickers)} tickers → {args.dump_tickers}")
        return

    # stale issuers (no filing in ~3y) carry outdated model ratings — comparing
    # them against a current agency view measures decay, not the scorecard
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=3)
    fresh = scored[scored["period_end"] >= cutoff]
    print(f"panel: {len(fresh)} of {len(scored)} issuers with a filing since "
          f"{cutoff.date()} (stale names excluded)\n")

    agency = load_agency_ratings(args.csv)
    print(format_report(compare_ratings(fresh, agency)))


if __name__ == "__main__":
    main()
