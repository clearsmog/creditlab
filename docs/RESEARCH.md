# Market & Landscape Research — Credit Risk Portfolio Project

*Synthesized 2026-07-12 from a multi-agent web research run: 5 search angles, 21 sources,
89 extracted claims, 25 adversarially verified (9 confirmed, 4 refuted, 12 unverified due
to a rate-limit failure — those are labelled). Confidence labels: **[C]** confirmed by
3-vote verification, **[U]** unverified but from a primary source, **[R]** refuted.*

## 1. What UK/EU employers actually screen for

- **[C]** The core deliverable set is PD, LGD, and EAD models under both **IFRS 9 and
  IRB** frameworks — job specs list this explicitly.
  ([Kingsgate job spec](https://kingsgaterecruitment.co.uk/job/credit-risk-manager-model-development-retail-irb-ifrs-9/))
- **[C]** Working knowledge of **PRA, EBA, and Basel 3.1** standards is expected.
  Basel 3.1 UK go-live is Jan 2027 and recruiters (Barclay Simpson 2025 guide) expect it
  to drive a credit-risk-modeller hiring surge — good timing for this project.
- **[C]** Methods demanded: **logistic regression, survival analysis, and machine
  learning** — the champion (scorecard) / challenger (ML) pattern maps directly onto job
  requirements. Survival analysis appears by name → include a survival-based lifetime PD
  component (it also feeds IFRS 9 lifetime ECL).
- **[R]** "Python+SAS+SQL is *the* UK stack" was refuted — specs accept SAS, Python, or R
  interchangeably. Python is a safe primary choice; demonstrate SQL in the data pipeline.
- Validation (calibration, back-testing, monitoring), stress testing, and
  ICAAP/capital-framework knowledge are named differentiators. Junior-to-mid pay band on
  the sampled spec: £40k–£90k; recruiter data: ~£40–45k graduate, £50–85k AVP (London).
- Market context: credit risk hiring softened in Q2 2025 with a tilt toward backfilling
  junior talent — structurally favorable for junior/mid candidates.

## 2. Competitive landscape — what differentiates

- The commodity tier is a **consumer-loan notebook PD scorecard** (Lending Club / HMEQ,
  WoE + logistic regression, Gini/AUC only, 0 stars). Hiring managers see these
  constantly.
- **Corporate credit is underrepresented**: fundamentals-based PD, Merton/KMV, transition
  matrices, and portfolio simulation are rare on GitHub — our chosen direction is the
  differentiated one.
- **[C]** The bar for "complete" open-source IFRS 9 work is
  [naenumtou/ifrs9](https://github.com/naenumtou/ifrs9) (~125 stars, notebooks, full
  PD/LGD/EAD impairment scope). Exceeding it means: corporate focus + validation
  documentation + a real interface. **[R]** "Only 10 IFRS 9 repos exist" was refuted
  (topic tags undercount), but the space is still thin relative to consumer scorecards.
- A recurring weak point to exploit: hobby projects claim "Basel-validated" with only
  Gini/AUC. A **full validation module** (calibration tests, PSI/CSI stability, CAP/AR,
  KS, out-of-time testing, SR 11-7-style write-up) is a genuine differentiator.
- Emerging pattern in 2025-26 repos: **platform-style delivery with a UI** rather than
  notebooks.
- Useful published benchmarks to beat/match:
  - Freddie Mac mortgage PD scorecard: AUC 0.83 / Gini 0.65 / KS 0.50 (out-of-time).
  - Consumer-loan champion/challenger: LR AUC 0.88 vs XGBoost 0.90 — the marginal gap
    *is* the story (interpretability costs little), with SHAP making the challenger
    defensible.

## 3. Tooling decisions

| Need | Choice | Why |
|---|---|---|
| Binning / WoE | **OptBinning** | The ecosystem foundation; native monotonic constraints; scorecardpy needs a separate add-on for monotonicity **[U]** |
| Scorecard pipeline | **skorecard** (ING Bank, MIT) | sklearn-compatible, wraps OptBinning, LR with p-values (regulatory doc need), ships a Dash bucket-tweaking app. Modest but real maintenance (v1.6.9, 2024) **[C]** |
| Pricing / counterparty | **ORE** — optional Phase 7 only | **[C]** ORE extends QuantLib; scope is XVA/counterparty, ~95% C++ with SWIG Python bindings **[U]**; BSD-licensed **[U]**; active (v1.8.16.0, May 2026). No IFRS 9 / scorecard / credit-portfolio features → confirms it must not be the foundation |
| Dashboard | **Streamlit + Plotly** | Free deploy (Community Cloud / HF Spaces), fastest solo iteration; Dash is the production-grade alternative (ING pairs scorecards with Dash) — acknowledge the tradeoff in the README. Rerun-model performance limits are acceptable for a demo |

## 4. Data strategy (revised — the biggest plan change)

License findings (all **[U]** — verification errored — but sourced from primary license
documents and a UK university libguide; treat as binding until checked):

- **LSEG/Refinitiv academic license**: research-only, ~10M data points/month,
  redistribution limited to "insubstantial portions", and **explicitly excludes use for
  employment opportunities**. A job-search portfolio built on Refinitiv data is a
  license violation risk → **avoid Refinitiv for this project**.
- **WRDS Compustat/CRSP**: internal research use; raw or reconstructable data cannot be
  published. Usable privately; publish only aggregate results.
- **SEC EDGAR XBRL API**: free, no key, effectively public-domain corporate fundamentals
  (10-K/10-Q), 10 req/s etiquette → **the license-clean backbone**. The ingestion
  pipeline itself becomes a demonstrable, fully reproducible feature.
- **S&P Annual Global Corporate Default & Rating Transition Study**: free, publicly
  citable default rates and transition matrices (1981–present) → calibrate PD term
  structures and transition matrices with zero licensing exposure. (Moody's DRD is the
  gold standard but not in standard university subscriptions.)
- **Freddie Mac SFLLD**: derived results may be published (noncommercial, must not allow
  dataset recreation); raw/derived loan-level data may not be redistributed. Retail
  mortgage data — optional LGD/EAD methodology module only.

**Revised strategy**: SEC EDGAR fundamentals + CRSP/market data (private) for the panel;
default/rating events and transition-matrix calibration from S&P/Moody's published
studies; WRDS as private enrichment with only aggregates published; no Refinitiv.

## 5. Dashboard visual vocabulary (from surveyed examples)

KPI tiles → portfolio EL/UL/economic capital; loss distribution with VaR/ES markers;
transition-matrix heatmap; default rate by rating band; sector/name concentration
(treemap or Lorenz/Gini curve); IFRS 9 stage migration waterfall; single-firm drill-down:
financials trend, ratio panel, fundamentals-PD vs Merton market-implied PD, SHAP
decomposition.

## Sources (main)

- https://www.barclaysimpson.com/salary-guides/risk-quants-salaries-and-recruitment-trends-2025/
- https://kingsgaterecruitment.co.uk/job/credit-risk-manager-model-development-retail-irb-ifrs-9/
- https://www.bartbaesens.com/course/85/credit-risk-modeling-for-basel-ifrs-9-using-r-python-sas
- https://github.com/ing-bank/skorecard · https://ing-bank.github.io/skorecard/
- https://opensourcerisk.org/ · https://github.com/OpenSourceRisk/Engine
- https://github.com/topics/ifrs-9 · https://github.com/naenumtou/ifrs9
- https://capitalmarkets.freddiemac.com/crt/docs/pdfs/fre_terms_conditions_sflld.pdf
- https://libanswers.citystgeorges.ac.uk/faq/281947 (LSEG academic license)
- https://guides.nyu.edu/wrds/faqs · https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- https://www.spglobal.com/ratings/en/regulatory/article/default-transition-and-recovery-2025-annual-global-corporate-default-and-rating-transition-study-s101673333
- https://quansight.com/post/dash-voila-panel-streamlit-our-thoughts-on-the-big-four-dashboarding-tools/
- https://github.com/Jane511/mortgage-credit-risk-pd-lgd-ead (benchmark exemplar)
