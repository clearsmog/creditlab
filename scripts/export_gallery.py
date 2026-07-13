"""Export static PNG gallery for README.

  uv run python scripts/export_gallery.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from creditlab.counterparty.exposure import pfe_addon
from creditlab.counterparty.limits import BASE_LIMIT_FRAC, HARD_CAP_USD, recommend_limit
from creditlab.models.scorecard import Scorecard, calibrate_pds
from creditlab.portfolio.ratings import GRADES, assign_rating
from creditlab.portfolio.simulation import simulate_losses, summarize
from creditlab.portfolio.transitions import SP_1Y, STATES

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
CENTRAL = 0.015


def theme(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        width=900,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK, family="system-ui, -apple-system, sans-serif", size=13),
        margin=dict(l=56, r=28, t=56, b=48),
        title=dict(text=title, font=dict(size=16, color=INK)) if title else None,
        legend=dict(orientation="h", y=1.08),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASE, tickfont=dict(color=MUTED))
    return fig


def save(fig: go.Figure, name: str) -> None:
    path = OUT / name
    fig.write_image(str(path), scale=2)
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")


def load_artifacts() -> dict:
    df = pd.read_parquet("data/processed/panel.parquet")
    df = df[df["period_end"] <= pd.Timestamp.today() - pd.DateOffset(years=1)]
    train = df[df["fyear"] <= 2019]
    card = Scorecard().fit(train, train["default_within_1y"])
    sample_rate = float(train["default_within_1y"].mean())
    df = df.assign(pd_cal=calibrate_pds(card.predict_pd(df), sample_rate, CENTRAL))
    df["rating"] = assign_rating(df["pd_cal"].to_numpy())
    latest = df.sort_values("period_end").groupby("cik").tail(1).copy()
    latest["ead"] = latest["liabilities"].clip(lower=0)
    latest = latest.dropna(subset=["ead", "equity"])
    book = latest[latest["rating"].isin(GRADES[:6])]  # drop CCC-heavy noise for viz
    losses = simulate_losses(book.rename(columns={"rating": "rating"}), n_sims=30_000)
    stats = summarize(losses, float(book["ead"].sum()))
    return {"panel": df, "latest": latest, "book": book, "losses": losses, "stats": stats}


def fig_rating_dist(book: pd.DataFrame) -> go.Figure:
    dist = book["rating"].value_counts().reindex(GRADES).dropna()
    fig = go.Figure(
        go.Bar(
            x=list(dist.index),
            y=dist.values,
            marker_color=BLUE,
            text=dist.values,
            textposition="outside",
            hovertemplate="%{x}: %{y} issuers<extra></extra>",
        )
    )
    return theme(fig, title="Issuers by internal rating (scorecard PD → master scale)")


def fig_loss_dist(losses: np.ndarray, stats: dict) -> go.Figure:
    fig = go.Figure(
        go.Histogram(
            x=losses / 1e9,
            nbinsx=100,
            marker_color=BLUE,
            hovertemplate="loss $%{x:.0f}bn · %{y} paths<extra></extra>",
        )
    )
    for label, key in [("EL", "EL"), ("VaR 99.9%", "VaR")]:
        v = stats[key] / 1e9
        fig.add_vline(
            x=v,
            line_dash="dash",
            line_color=INK2,
            annotation_text=f"{label} ${v:.0f}bn",
            annotation_position="top right",
        )
    fig.update_layout(yaxis_type="log", xaxis_title="portfolio loss ($bn)", yaxis_title="paths (log)")
    return theme(fig, height=440, title="Portfolio loss distribution (Monte Carlo)")


def fig_transitions() -> go.Figure:
    m = SP_1Y * 100
    fig = go.Figure(
        go.Heatmap(
            z=m,
            x=STATES,
            y=STATES,
            colorscale=[[0, SEQ[0]], [0.5, SEQ[3]], [1, SEQ[6]]],
            zmin=0,
            zmax=100,
            text=np.round(m, 1),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="%"),
            hovertemplate="%{y} → %{x}: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", title="from")
    fig.update_xaxes(side="top", title="to")
    return theme(fig, height=480, title="One-year rating transition matrix (%)")


def fig_firm_pd(panel: pd.DataFrame) -> go.Figure:
    # pick a name with multi-year history and equity
    latest = panel.sort_values("period_end").groupby("cik").tail(1)
    cand = latest.dropna(subset=["equity", "ticker"])
    cand = cand[cand["equity"] > 1e9]
    row = cand.sort_values("pd_cal").iloc[len(cand) // 3]
    firm = panel[panel["cik"] == row["cik"]].sort_values("period_end")
    name = str(row.get("name") or row.get("ticker") or "issuer")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=firm["fyear"],
            y=firm["pd_cal"] * 100,
            name="1y PD (%)",
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(size=8),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=firm["fyear"],
            y=firm["leverage"],
            name="Leverage (TL/TA)",
            mode="lines+markers",
            line=dict(color=AQUA, width=2),
            marker=dict(size=8),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="PD (%)", secondary_y=False)
    fig.update_yaxes(title_text="Leverage", secondary_y=True)
    return theme(fig, title=f"Counterparty drill-down — {name[:48]}")


def fig_limit_grid() -> go.Figure:
    grades = list(BASE_LIMIT_FRAC)
    fracs = [BASE_LIMIT_FRAC[g] * 100 for g in grades]
    caps = [HARD_CAP_USD[g] / 1e6 for g in grades]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=grades, y=fracs, name="% of equity", marker_color=BLUE),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=grades,
            y=caps,
            name="Hard cap ($m)",
            mode="lines+markers",
            line=dict(color=YELLOW, width=2),
            marker=dict(size=9),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Base limit as % of equity", secondary_y=False)
    fig.update_yaxes(title_text="Hard cap ($m)", secondary_y=True)
    return theme(fig, title="Demo unsecured limit grid by rating grade")


def fig_desk_headroom(latest: pd.DataFrame) -> go.Figure:
    # score a handful of names and show limit vs PFE for a fixed ticket
    sample = latest.dropna(subset=["equity", "ticker"]).copy()
    sample = sample[sample["equity"] > 5e8].sort_values("pd_cal").head(12)
    rows = []
    for _, row in sample.iterrows():
        rec = recommend_limit(row, str(row["rating"]), float(row["pd_cal"]))
        pfe = pfe_addon(15e6, 1.0)
        rows.append(
            {
                "name": str(row.get("name") or row["ticker"])[:28],
                "limit": rec.recommended_limit_usd / 1e6,
                "pfe": pfe / 1e6,
                "rating": rec.rating,
            }
        )
    d = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Proposed limit ($m)", x=d["name"], y=d["limit"], marker_color=BLUE)
    )
    fig.add_trace(
        go.Scatter(
            name="Illustrative deal PFE ($m)",
            x=d["name"],
            y=d["pfe"],
            mode="lines+markers",
            line=dict(color=YELLOW, width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(xaxis_tickangle=-35, yaxis_title="$m")
    return theme(fig, height=460, title="Desk view: limit vs $15m × 1y PFE add-on (demo ticket)")


def fig_ratio_radar(latest: pd.DataFrame) -> go.Figure:
    """Simple multi-metric comparison for two issuers."""
    sample = latest.dropna(subset=["leverage", "interest_coverage", "current_ratio", "roa", "pd_cal"])
    sample = sample[sample["equity"] > 1e9].sort_values("pd_cal")
    lo, hi = sample.iloc[0], sample.iloc[min(20, len(sample) - 1)]

    def vec(r: pd.Series) -> list[float]:
        # invert leverage so higher = safer; clip for display
        return [
            float(np.clip(1 - r["leverage"], 0, 1)),
            float(np.clip(r["interest_coverage"] / 10, 0, 1)),
            float(np.clip(r["current_ratio"] / 2, 0, 1)),
            float(np.clip((r["roa"] + 0.05) / 0.15, 0, 1)),
            float(np.clip(1 - r["pd_cal"] * 20, 0, 1)),
        ]

    cats = ["Solvency", "Coverage", "Liquidity", "Profitability", "Low PD"]
    fig = go.Figure()
    for r, color in ((lo, BLUE), (hi, AQUA)):
        v = vec(r)
        fig.add_trace(
            go.Scatterpolar(
                r=v + v[:1],
                theta=cats + cats[:1],
                fill="toself",
                name=str(r.get("name") or r.get("ticker"))[:32],
                line=dict(color=color),
            )
        )
    fig.update_layout(
        polar=dict(
            bgcolor=SURFACE,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID),
            angularaxis=dict(gridcolor=GRID),
        )
    )
    return theme(fig, height=460, title="Counterparty profile — safer vs weaker issuer (normalized)")


def main() -> None:
    a = load_artifacts()
    save(fig_rating_dist(a["book"]), "01-rating-distribution.png")
    save(fig_firm_pd(a["panel"]), "02-counterparty-pd-leverage.png")
    save(fig_limit_grid(), "03-limit-grid.png")
    save(fig_desk_headroom(a["latest"]), "04-limit-vs-pfe.png")
    save(fig_loss_dist(a["losses"], a["stats"]), "05-portfolio-loss.png")
    save(fig_transitions(), "06-transition-matrix.png")
    save(fig_ratio_radar(a["latest"]), "07-counterparty-radar.png")
    print(f"\ngallery ready under {OUT}/")


if __name__ == "__main__":
    main()
