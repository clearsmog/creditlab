# CreditLab

An end-to-end corporate credit risk platform: PD/LGD/EAD modeling from real corporate
financial data, IFRS 9 ECL, structural models, portfolio credit risk simulation, and an
interactive dashboard.

Built by an FRM charterholder as a working demonstration of the full credit risk model
lifecycle — development, calibration, and validation — following industry practice
(scorecard development standards, IFRS 9 methodology, SR 11-7 / PRA-style validation).

## Architecture

```
data sources (SEC EDGAR fundamentals, CRSP/WRDS market data,
              S&P/Moody's published default & transition studies)
        |
        v
[1] data pipeline ──> firm-year panel with ratios + default/rating events
        |
        v
[2] PD models ─────> Altman Z benchmark │ logistic scorecard │ ML challenger + SHAP
        |
        v
[3] structural ────> Merton / KMV distance-to-default (market-implied PD)
        |
        v
[4] ratings layer ─> master scale mapping │ transition matrices │ LGD assumptions
        |
        v
[5] portfolio ─────> Vasicek / CreditMetrics Monte Carlo │ economic capital │ IFRS 9 ECL + stress
        |
        v
[6] dashboard ─────> Streamlit: portfolio view, loss distributions, firm drill-down
```

## Package layout

| Module | Contents |
|---|---|
| `creditlab.data` | Data ingestion, panel construction, ratio engineering |
| `creditlab.models` | PD models (scorecard, ML challenger), Merton/KMV, validation metrics |
| `creditlab.portfolio` | Transition matrices, Vasicek/CreditMetrics simulation, economic capital |
| `creditlab.ecl` | IFRS 9 staging, PD term structures, scenario-weighted ECL, stress testing |
| `creditlab.viz` | Reusable Plotly chart builders for the dashboard and reports |

## Roadmap

- [ ] **Phase 1 — Data pipeline**: firm-year panel from SEC EDGAR XBRL fundamentals
      (license-clean, fully reproducible) enriched privately with WRDS Compustat/CRSP;
      default and rating events; financial ratio engineering
- [ ] **Phase 2 — PD models**: Altman Z benchmark, WoE/IV logistic scorecard
      (OptBinning/skorecard), gradient boosting challenger with SHAP, survival-analysis
      lifetime PD; full validation suite (AUC/Gini/KS/CAP, calibration tests, PSI
      out-of-time stability) with an SR 11-7-style validation report per model
- [ ] **Phase 3 — Structural models**: Merton distance-to-default; market-implied vs
      fundamentals-based PD comparison
- [ ] **Phase 4 — Ratings layer**: PD-to-rating master scale, transition matrices
      calibrated to S&P/Moody's published default & transition studies, LGD from rating
      agency recovery data
- [ ] **Phase 5 — Portfolio risk**: Vasicek single-factor and CreditMetrics Monte Carlo
      loss distribution, economic capital, IFRS 9 ECL with macro scenario weighting
- [ ] **Phase 6 — Dashboard**: interactive Streamlit app
- [ ] **Phase 7 (optional) — Counterparty risk**: CVA/PFE demo via ORE Python bindings

## Data licensing

Models are developed on licensed academic data (WRDS Compustat/CRSP primarily, with
LSEG/Bloomberg where useful), accessed legitimately through a university subscription.
License terms prohibit redistributing raw or reconstructable data, so this repo publishes
code, methodology, and aggregate results only — never the underlying panel.

For reproducibility, the pipeline also supports the SEC EDGAR XBRL API — free and
license-clean — so anyone can run the full platform end-to-end on public data without a
data subscription. Transition matrices and default-rate calibration come from S&P/Moody's
publicly published annual default & transition studies, which are citable without
restriction. See `docs/RESEARCH.md` for details.

## Setup

```sh
uv sync
```
