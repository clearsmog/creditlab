"""CreditLab dashboard.

Run:  uv run streamlit run src/creditlab/dashboard.py
Pages are addressable via ?page=<overview|portfolio|transitions|ifrs9|firm>.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from creditlab.ecl.engine import SCENARIOS, ecl, stage_of, weighted_ecl
from creditlab.models.scorecard import Scorecard, calibrate_pds
from creditlab.portfolio.ratings import GRADES, assign_rating
from creditlab.portfolio.simulation import simulate_losses, summarize
from creditlab.portfolio.transitions import SP_1Y, STATES, cumulative_pd

# palette (dataviz reference instance, light mode)
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"

CENTRAL_TENDENCY = 0.015


def themed(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK, family="system-ui, -apple-system, sans-serif"),
        margin=dict(l=48, r=24, t=36, b=40), hovermode="closest",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=BASE, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=BASE, tickfont=dict(color=MUTED))
    return fig


@st.cache_resource(show_spinner="fitting models and simulating…")
def artifacts() -> dict:
    df = pd.read_parquet("data/processed/panel.parquet")
    df = df[df["period_end"] <= pd.Timestamp.today() - pd.DateOffset(years=1)]
    train = df[df["fyear"] <= 2019]
    card = Scorecard().fit(train, train["default_within_1y"])
    sample_rate = train["default_within_1y"].mean()

    df = df.assign(
        pd_cal=calibrate_pds(card.predict_pd(df), sample_rate, CENTRAL_TENDENCY)
    )
    df["rating"] = assign_rating(df["pd_cal"])

    latest = df.sort_values("period_end").groupby("cik").tail(1).copy()
    first = df.sort_values("period_end").groupby("cik").head(1)
    latest = latest.merge(
        first[["cik", "rating"]].rename(columns={"rating": "rating_orig"}), on="cik"
    )
    latest["ead"] = latest["liabilities"].clip(lower=0)
    latest = latest.dropna(subset=["ead"])
    latest["stage"] = [
        stage_of(n, o, bool(d)) for n, o, d in
        zip(latest["rating"], latest["rating_orig"], latest["default_within_1y"])
    ]
    latest["ecl"] = [weighted_ecl(r, e, s) for r, e, s in
                     zip(latest["rating"], latest["ead"], latest["stage"])]

    book = latest[latest["stage"] < 3]
    losses = simulate_losses(book.rename(columns={"rating": "rating"}), n_sims=50_000)
    stats = summarize(losses, float(book["ead"].sum()))
    return {"panel": df, "latest": latest, "book": book, "losses": losses, "stats": stats}


def page_overview(a: dict) -> None:
    st.subheader("Portfolio overview")
    book, s = a["book"], a["stats"]
    c = st.columns(5)
    c[0].metric("Obligors", f"{len(book):,}")
    c[1].metric("EAD", f"${s['total_ead']/1e12:.2f}tn")
    c[2].metric("Expected loss", f"{s['EL']/s['total_ead']:.2%}")
    c[3].metric("VaR 99.9%", f"{s['VaR']/s['total_ead']:.1%}")
    c[4].metric("IFRS 9 ECL", f"${a['latest']['ecl'].sum()/1e9:.0f}bn")

    dist = book["rating"].value_counts().reindex(GRADES).dropna()
    fig = go.Figure(go.Bar(
        x=dist.index, y=dist.values, marker_color=BLUE,
        marker_line_color=SURFACE, marker_line_width=2,
        text=dist.values, textposition="outside", textfont=dict(color=INK2),
        hovertemplate="%{x}: %{y} obligors<extra></extra>",
    ))
    fig.update_layout(title="Obligors by rating grade", showlegend=False)
    st.plotly_chart(themed(fig), use_container_width=True)


def page_portfolio(a: dict) -> None:
    st.subheader("Loss distribution — one-factor Monte Carlo (50k paths)")
    losses, s = a["losses"] / 1e9, a["stats"]
    fig = go.Figure(go.Histogram(
        x=losses, nbinsx=120, marker_color=BLUE,
        marker_line_color=SURFACE, marker_line_width=1,
        hovertemplate="loss $%{x:.0f}bn: %{y} paths<extra></extra>",
    ))
    for label, v, pos in [("EL", s["EL"], "top right"),
                          ("VaR 99.9%", s["VaR"], "top left"),
                          ("ES", s["ES"], "top right")]:
        fig.add_vline(x=v / 1e9, line_dash="dash", line_color=INK2,
                      annotation_text=f"{label} ${v/1e9:.0f}bn",
                      annotation_position=pos, annotation_font_color=INK2)
    fig.update_layout(yaxis_type="log", xaxis_title="portfolio loss ($bn)",
                      yaxis_title="paths (log)", showlegend=False)
    st.plotly_chart(themed(fig, 420), use_container_width=True)

    top = a["book"].nlargest(12, "ead")[["name", "rating", "ead"]]
    top["ead"] = (top["ead"] / 1e9).round(1)
    st.caption("Largest exposures ($bn) — single-name concentration drives the "
               "gap between simulated capital and summed IRB charges.")
    st.dataframe(top, hide_index=True, use_container_width=True)


def page_transitions() -> None:
    st.subheader("S&P average one-year transition matrix (%)")
    m = SP_1Y * 100
    fig = go.Figure(go.Heatmap(
        z=m, x=STATES, y=STATES, colorscale=[[0, SEQ[0]], [0.5, SEQ[3]], [1, SEQ[6]]],
        zmin=0, zmax=100, text=np.round(m, 2), texttemplate="%{text}",
        textfont=dict(size=11), hovertemplate="%{y} → %{x}: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="%", outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed", title="from")
    fig.update_xaxes(side="top", title=None)
    st.plotly_chart(themed(fig, 460), use_container_width=True)

    st.subheader("Cumulative PD term structures (Markov)")
    horizons = list(range(1, 11))
    fig = go.Figure()
    for grade, color in [("BBB", BLUE), ("BB", AQUA), ("B", YELLOW)]:
        curve = cumulative_pd(grade, horizons) * 100
        fig.add_trace(go.Scatter(
            x=horizons, y=curve, name=grade, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"{grade} %{{x}}y: %{{y:.2f}}%<extra></extra>",
        ))
        fig.add_annotation(x=horizons[-1], y=float(curve.iloc[-1]), text=grade,
                           xanchor="left", showarrow=False, font=dict(color=color))
    fig.update_layout(xaxis_title="horizon (years)", yaxis_title="cumulative PD (%)",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(themed(fig), use_container_width=True)
    st.caption("Matrix powers overstate long-horizon speculative-grade PDs vs "
               "published cumulatives — within-grade heterogeneity beats momentum.")


def page_ifrs9(a: dict) -> None:
    st.subheader("IFRS 9 staging")
    latest = a["latest"]
    agg = latest.groupby("stage").agg(
        obligors=("cik", "size"), ead=("ead", "sum"), ecl=("ecl", "sum"))
    agg["coverage"] = agg["ecl"] / agg["ead"]

    c = st.columns(3)
    for i, (stage, row) in enumerate(agg.iterrows()):
        c[i].metric(f"Stage {stage}", f"{int(row.obligors)} obligors",
                    f"coverage {row.coverage:.1%}", delta_color="off")

    fig = go.Figure(go.Bar(
        x=[f"Stage {i}" for i in agg.index], y=agg["coverage"] * 100,
        marker_color=BLUE, marker_line_color=SURFACE, marker_line_width=2,
        text=[f"{v:.1%}" for v in agg["coverage"]], textposition="outside",
        textfont=dict(color=INK2), hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(title="ECL coverage by stage", yaxis_title="ECL / EAD (%)",
                      showlegend=False)
    st.plotly_chart(themed(fig), use_container_width=True)

    st.subheader("Scenario decomposition")
    perf = latest[latest["stage"] < 3]
    rows = [(name, sum(ecl(r, e, s, z=z) for r, e, s in
                       zip(perf["rating"], perf["ead"], perf["stage"])) / 1e9)
            for name, z, _ in SCENARIOS]
    sc = pd.DataFrame(rows, columns=["scenario", "ecl_bn"])
    fig = go.Figure(go.Bar(
        x=sc["scenario"], y=sc["ecl_bn"], marker_color=BLUE,
        marker_line_color=SURFACE, marker_line_width=2,
        text=[f"${v:.1f}bn" for v in sc["ecl_bn"]], textposition="outside",
        textfont=dict(color=INK2), hovertemplate="%{x}: $%{y:.1f}bn<extra></extra>",
    ))
    fig.update_layout(title="Performing-book ECL by macro scenario "
                            "(systematic factor: +1.28 / 0 / −1.28)",
                      yaxis_title="ECL ($bn)", showlegend=False)
    st.plotly_chart(themed(fig), use_container_width=True)


def page_firm(a: dict) -> None:
    st.subheader("Firm explorer")
    panel = a["panel"]
    tickers = sorted(t for t in panel["ticker"].unique() if t)
    ticker = st.selectbox("issuer", tickers)
    firm = panel[panel["ticker"] == ticker].sort_values("period_end")
    last = firm.iloc[-1]

    c = st.columns(4)
    c[0].metric("Rating", str(last["rating"]))
    c[1].metric("Calibrated 1y PD", f"{last['pd_cal']:.2%}")
    c[2].metric("Altman Z'", "n/a" if pd.isna(last["altman_z_prime"])
                else f"{last['altman_z_prime']:.2f}")
    c[3].metric("Defaulted", "yes" if firm["default_within_1y"].any() else "no")

    pairs = [("leverage", "Leverage (TL/TA)"), ("interest_coverage", "Interest coverage"),
             ("roa", "Return on assets"), ("current_ratio", "Current ratio")]
    cols = st.columns(2)
    for i, (col, title) in enumerate(pairs):
        fig = go.Figure(go.Scatter(
            x=firm["fyear"], y=firm[col], mode="lines+markers",
            line=dict(color=BLUE, width=2), marker=dict(size=8),
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(title=title, showlegend=False)
        cols[i % 2].plotly_chart(themed(fig, 260), use_container_width=True)

    fig = go.Figure(go.Scatter(
        x=firm["fyear"], y=firm["pd_cal"] * 100, mode="lines+markers",
        line=dict(color=BLUE, width=2), marker=dict(size=8),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(title="Calibrated scorecard PD history", yaxis_title="PD (%)",
                      showlegend=False)
    st.plotly_chart(themed(fig, 300), use_container_width=True)


PAGES = {
    "overview": ("Overview", page_overview),
    "portfolio": ("Portfolio risk", page_portfolio),
    "transitions": ("Transitions", lambda a: page_transitions()),
    "ifrs9": ("IFRS 9", page_ifrs9),
    "firm": ("Firm explorer", page_firm),
}


def main() -> None:
    st.set_page_config(page_title="CreditLab", layout="wide")
    st.title("CreditLab — corporate credit risk")

    keys = list(PAGES)
    default = st.query_params.get("page", "overview")
    idx = keys.index(default) if default in keys else 0
    choice = st.sidebar.radio("Section", keys, index=idx,
                              format_func=lambda k: PAGES[k][0])
    st.query_params["page"] = choice
    PAGES[choice][1](artifacts())


main()
