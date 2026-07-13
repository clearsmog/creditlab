# CreditLab

End-to-end **corporate credit risk** platform: financial-statement panel construction, PD modeling (scorecard + ML challenger + Merton), ratings / LGD, portfolio credit risk, IFRS 9 ECL, and an interactive Streamlit dashboard.

Built as a working demonstration of the credit model lifecycle — development, calibration, and validation — aligned with industry practice (scorecard standards, IFRS 9 methodology, model-validation hygiene).

> **Portfolio project.** Educational / research use. Not a production risk system and not investment advice.

---

## Features

| Area | What it does |
| --- | --- |
| **Data pipeline** | SEC EDGAR XBRL fundamentals → firm-year panel, financial ratios, default/rating labels |
| **PD models** | Altman Z benchmark, WoE/IV logistic scorecard, gradient-boosting challenger, validation metrics (AUC/Gini/KS/calibration) |
| **Structural PD** | Merton / KMV-style distance-to-default (market-implied PD) |
| **Ratings & LGD** | Master-scale mapping, transition matrices (S&P/Moody’s published studies), recovery assumptions |
| **Portfolio risk** | Vasicek / CreditMetrics-style Monte Carlo, loss distribution, economic capital intuition |
| **IFRS 9 ECL** | Staging, scenario-weighted ECL, simple stress views |
| **Dashboard** | Streamlit UI: portfolio overview, loss distribution, transitions, IFRS 9, firm explorer |

---

## Architecture

```
data sources
  SEC EDGAR fundamentals (+ optional private WRDS Compustat/CRSP)
  S&P / Moody’s published default & transition studies
        │
        ▼
[1] data pipeline ──► firm-year panel (ratios + events)
        │
        ▼
[2] PD models ──────► Altman Z │ logistic scorecard │ ML challenger
        │
        ▼
[3] structural ─────► Merton distance-to-default
        │
        ▼
[4] ratings layer ──► master scale │ transitions │ LGD
        │
        ▼
[5] portfolio ──────► Vasicek / Monte Carlo │ economic capital │ IFRS 9 ECL
        │
        ▼
[6] dashboard ──────► Streamlit (overview · portfolio · firm drill-down)
```

---

## Repository layout

```
Credit/
├── src/creditlab/
│   ├── data/          # EDGAR ingest, universe, ratios, labels
│   ├── models/        # scorecard, challenger, Merton, metrics, binning
│   ├── portfolio/     # ratings, transitions, LGD, Vasicek, simulation
│   ├── ecl/           # IFRS 9 ECL engine
│   ├── viz/           # chart helpers
│   └── dashboard.py   # Streamlit app
├── tests/             # pytest suite
├── docs/              # research notes
├── data/              # local only (gitignored): raw/ + processed/
├── pyproject.toml
└── README.md
```

| Package | Role |
| --- | --- |
| `creditlab.data` | Ingestion, panel construction, ratio engineering |
| `creditlab.models` | PD models, Merton, validation metrics |
| `creditlab.portfolio` | Transitions, simulation, economic capital |
| `creditlab.ecl` | IFRS 9 staging & scenario-weighted ECL |
| `creditlab.viz` | Plotly builders for dashboard / reports |

---

## Requirements

- Python **3.12+**
- [`uv`](https://github.com/astral-sh/uv) (recommended)

Raw EDGAR pulls need network access and a polite User-Agent (configured in the EDGAR client). Large caches live under `data/` and are **not** committed.

---

## Setup

```sh
git clone https://github.com/clearsmog/creditlab.git
cd creditlab
uv sync
```

Install with dev/test tools:

```sh
uv sync --group dev
```

---

## Quick start

### Run tests

```sh
uv run pytest -q
```

### Fit / demo individual modules

Many modules expose a small `main()` for smoke demos:

```sh
uv run python -m creditlab.models.scorecard
uv run python -m creditlab.models.challenger
uv run python -m creditlab.models.merton
uv run python -m creditlab.ecl.engine
uv run python -m creditlab.portfolio.vasicek
```

### Streamlit dashboard

```sh
uv run streamlit run src/creditlab/dashboard.py
```

Pages (also via query string): `overview` · `portfolio` · `transitions` · `ifrs9` · `firm`

Example: `http://localhost:8501/?page=firm`

### Build / refresh the data panel

Requires network for EDGAR (and optional WRDS credentials if you enrich privately):

```sh
uv run python -m creditlab.data.universe --help
```

Processed outputs are written under `data/processed/` (gitignored).

---

## Data & licensing

| Source | Use in this project |
| --- | --- |
| **SEC EDGAR XBRL API** | Primary, license-clean fundamentals backbone (public) |
| **WRDS Compustat / CRSP** | Optional private enrichment (university license; **do not redistribute**) |
| **S&P / Moody’s published default & transition studies** | Calibration references (public annual studies) |

- Raw and processed data under `data/` are **gitignored**.
- Do not publish WRDS-derived reconstructable datasets.
- Respect SEC fair-access etiquette (identify User-Agent; stay under rate limits).

---

## Roadmap

- [x] **Phase 1** — Data pipeline (EDGAR firm-year panel, ratios, labels)
- [x] **Phase 2** — PD models (scorecard, ML challenger, validation metrics)
- [x] **Phase 3** — Structural models (Merton DtD)
- [x] **Phase 4** — Ratings layer (master scale, transitions, LGD)
- [x] **Phase 5** — Portfolio risk + IFRS 9 ECL
- [x] **Phase 6** — Streamlit dashboard
- [ ] **Phase 7 (optional)** — Counterparty credit demo (e.g. CVA/PFE via ORE)

---

## Design notes

- **Corporate** focus (fundamentals + structural PD), not a consumer-loan scorecard notebook.
- **Champion / challenger**: interpretable scorecard + ML model with validation diagnostics.
- **Reproducible core path** on public EDGAR data; optional private market data stays local.
- Tests cover metrics, binning, EDGAR fixtures, Merton round-trips, portfolio helpers.

---

## Disclaimer

This repository is for **learning, interview portfolio, and research illustration**. Model outputs are not credit ratings, not regulatory-approved methodologies, and must not be used for live lending, trading, or capital decisions without independent validation and governance.

---

## Author

**Qiankun (Kenny) Zhu** · FRM · [GitHub](https://github.com/clearsmog) · [LinkedIn](https://www.linkedin.com/in/kenny0908)

---

## License

No license file is attached yet. Treat as **all rights reserved** unless a `LICENSE` is added (e.g. MIT). Contact the author before commercial reuse.
