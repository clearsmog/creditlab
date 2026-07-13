# CreditLab

**Trading-credit desk toolkit** for energy / commodity markets, built on a full corporate credit lab.

Primary workflow (matches energy-merchant **Credit Risk Analyst** work):

1. **Counterparty assessment** from financial statements (ratio flags + model PD / internal rating)  
2. **Unsecured limit recommendation** (transparent rating grid + haircuts + max tenor)  
3. **Documentation / KYC cues** (ISDA, CSA, PCG, monitoring intensity)  
4. **Pre-deal check** — simple PFE-style add-on vs limit headroom  
5. **FO credit memo** — plain-language opinion a trading desk can action  

Under the hood: SEC EDGAR firm-year panel, scorecard + ML PD models, Merton DtD, ratings / transitions, portfolio Monte Carlo, and IFRS 9 ECL — available as lab modules.

> **Portfolio project.** Educational / research use. Limit policy is pedagogical, not a live house methodology. Not investment advice.

---

## Why this shape (trading credit, not bank IRB)

| Energy trading credit desk needs | CreditLab |
| --- | --- |
| Analyse counterparties via financials → **limits** | Scorecard PD + ratio flags + limit engine |
| FO partnership — clear yes / no / structure | FO memo generator |
| Trading docs (ISDA, CSA, PCG) | Doc pack by risk grade |
| Exposure vs limit (MtM / PFE intuition) | PFE add-on + headroom check |
| KYC status discipline | Demo KYC traffic light (clear / review / escalate) |
| Portfolio capital / IFRS 9 (secondary) | Lab pages still available |

---

## Quick start

```sh
git clone https://github.com/clearsmog/creditlab.git
cd creditlab
uv sync --group dev
```

### Trading credit desk (CLI)

Requires local `data/processed/panel.parquet` (build via EDGAR pipeline, or use your existing panel):

```sh
uv run python -m creditlab.counterparty.desk
uv run python -m creditlab.counterparty.desk --ticker XOM --notional 25000000 --tenor 1.5
```

### Dashboard (default page = Trading credit desk)

```sh
uv run streamlit run src/creditlab/dashboard.py
```

| Page | Purpose |
| --- | --- |
| **Trading credit desk** | Counterparty → limit → FO memo → pre-deal utilisation |
| Firm explorer | Ratio & PD history |
| Portfolio overview / risk | Book composition, Monte Carlo losses |
| Transitions | S&P-style matrix & cumulative PDs |
| IFRS 9 | Staging & scenario ECL (lab) |

### Tests

```sh
uv run pytest -q
```

---

## Trading-credit module

```
src/creditlab/counterparty/
├── limits.py      # ratio flags, rating→limit grid, doc packs
├── exposure.py    # PFE add-on proxy, limit headroom
├── memo.py        # FO-facing markdown memo
└── desk.py        # CLI end-to-end demo
```

### Limit policy (demo)

Transparent construction (replace with house policy in production):

1. Map model **1y PD → rating grade** (S&P-anchored master scale)  
2. **Base capacity** = min(equity × grade fraction, hard cap)  
3. Apply **haircuts** from leverage / coverage / liquidity / ROA flags  
4. Attach **max tenor** and **documentation pack** by grade  

CCC / weak names → **no unsecured line** (prepay / LC / full CSA).

### Pre-deal exposure (demo)

```
PFE_addon ≈ notional × σ × √T × z
```

Default σ = 35% (energy-ish placeholder). Use for *conversation*, not VaR sign-off.

---

## Architecture (full lab)

```
SEC EDGAR (+ optional private WRDS)
        │
        ▼
[1] data pipeline ──► firm-year panel (ratios + default labels)
        │
        ▼
[2] PD models ──────► Altman Z │ logistic scorecard │ ML challenger
        │
        ▼
[3] structural ─────► Merton distance-to-default
        │
        ▼
[4] ratings ────────► master scale │ transitions │ LGD
        │
        ├──────────────────────────────┐
        ▼                              ▼
[5a] TRADING DESK                 [5b] LAB
  limits · docs · FO memo           portfolio MC · IFRS 9 ECL
  PFE vs limit headroom
        │
        ▼
[6] Streamlit dashboard (desk-first navigation)
```

| Package | Role |
| --- | --- |
| `creditlab.counterparty` | **Trading desk:** limits, exposure, FO memo |
| `creditlab.data` | EDGAR ingest, panel, ratios, labels |
| `creditlab.models` | PD models, Merton, validation metrics |
| `creditlab.portfolio` | Transitions, simulation, economic capital |
| `creditlab.ecl` | IFRS 9 staging & scenario ECL |
| `creditlab.viz` | Plotly helpers |

---

## Data & licensing

| Source | Use |
| --- | --- |
| **SEC EDGAR XBRL API** | Primary fundamentals (public, license-clean) |
| **WRDS Compustat / CRSP** | Optional private enrichment — **do not redistribute** |
| **S&P / Moody’s published studies** | Transition / default-rate anchors |

`data/raw/` and `data/processed/` are **gitignored**. Respect SEC rate limits and User-Agent rules.

---

## Roadmap

- [x] Corporate panel + PD models + Merton + ratings  
- [x] Portfolio MC + IFRS 9 ECL + dashboard  
- [x] **Trading credit desk** (limits, PFE check, FO memo)  
- [ ] Energy sector peer sets / commodity offtaker templates  
- [ ] Optional CVA/PFE via ORE (true counterparty risk)  
- [ ] Export limit blotter to CSV for “Credit Risk Cube”-style ops demos  

---

## Disclaimer

Illustrative only. Not a regulatory model, not a real credit decision, and not affiliated with SEFE or any trading house. Do not use for live lending, trading limits, or capital without independent validation and governance.

---

## Author

**Qiankun (Kenny) Zhu** · FRM · [GitHub](https://github.com/clearsmog) · [LinkedIn](https://www.linkedin.com/in/kenny0908)

---

## License

All rights reserved until a `LICENSE` file is added.
