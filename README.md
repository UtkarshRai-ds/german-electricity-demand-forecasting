# ⚡ German Electricity Demand Forecasting — SMARD / Bundesnetzagentur

**Forecasting Germany's national electricity demand 24 hours ahead and beating the grid operator's own day-ahead forecast on the same period using time series analysis.**

🔗 **Live:** https://german-electricity-demand-forecasting.streamlit.app/
📊 **Dataset:** [SMARD.de](https://www.smard.de/en) (Bundesnetzagentur) · [Open-Meteo](https://open-meteo.com)

---

## 1 · Project Overview

**The problem.** Germany's transmission system operators (TSOs) must predict national
electricity demand a day in advance so generators schedule the right amount of power.
Under-forecast and the grid risks shortfalls; over-forecast and fuel, money, and emissions
are wasted. Even a fraction of a percentage point of accuracy translates into meaningful
operational savings across a ~60,000 MWh national load.

**Who uses it.** The intended end user is a grid analyst or energy trader who needs an
accurate, reproducible day-ahead load forecast.

**The data.** Six years of hourly German grid load (2020–2026) from SMARD.de, joined with
hourly temperature from Open-Meteo. After lag-warmup the modelling set is ~56,000 hourly rows.

**What the model outputs.** A point forecast of national grid load (MWh) for each hour, 24
hours ahead, using only information that would genuinely be available at forecast time.

**Key design decision.** The whole project is built around a **strict 24-hour forecast
horizon and a leakage audit.** Every feature derived from load or temperature is shifted so
the model never sees information from the hour it is predicting. This is the difference between
a model that *looks* accurate and one that *is* and it is documented openly in
[`reports/leak_audit.md`](reports/leak_audit.md).

---

## 2 · Headline Result

On a strict hold-out of the first half of 2026 the data and model never saw in training. The
best model (**CatBoost**) achieves **2.83% MAPE**, versus the official TSO day-ahead
forecast's **4.40% MAPE** computed on the *identical* hours.

> **CatBoost is ~36% more accurate than the official grid-operator forecast on the same 2026 hold-out.**

The honest 2.83% is reported *after* a feature-leakage fix raised it from an inflated 2.24%
see [Key Decisions & Lessons](#9--key-decisions--lessons).

---

## 3 · Architecture

```
┌──────────────┐     ┌──────────────┐
│  SMARD.de    │     │  Open-Meteo  │
│ hourly load  │     │   weather    │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    ▼
   ┌─────────────────────────────┐
   │  Data Loading & Cleaning    │   src/data/  (loader, cleaner,
   │  + quality checks           │              quality, weather)
   └──────────────┬──────────────┘
                  ▼
   ┌─────────────────────────────┐
   │  Feature Engineering        │   src/features/engineer.py
   │  calendar · lag · rolling · │   ── horizon-shifted to
   │  temperature                │      prevent target leakage
   └──────────────┬──────────────┘
                  ▼
   ┌─────────────────────────────┐
   │  Leakage Audit (24h shift)  │   reports/leak_audit.md
   └──────────────┬──────────────┘
                  ▼
   ┌─────────────────────────────┐
   │  Six Models                 │   src/models/
   │  Seasonal-Naive · SARIMA ·  │   baseline · sarima ·
   │  Prophet · LightGBM ·       │   prophet · boosting
   │  XGBoost · CatBoost         │
   └──────────────┬──────────────┘
            │             │
            ▼             ▼
   ┌──────────────┐  ┌──────────────────┐
   │   MLflow     │  │  Best model .pkl │
   │  tracking    │  │  + export_for_app│
   └──────────────┘  └────────┬─────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Streamlit Dashboard│   app/streamlit_app.py
                   │  forecast vs actual │
                   │  vs TSO · leak audit│
                   └─────────────────────┘
```

---

## 4 · Results

All models tested on the same 2026 hold-out (`2026-01-01` → `2026-06-15`).

| Model | MAPE | Tested on | Notes |
|---|---|---|---|
| Seasonal-Naive (lag-168) | 6.44% | hourly | Simple "same hour last week" baseline |
| SARIMA(2,1,2)(1,0,1,7) | 36.96% | daily | Diverged — structural breaks (see lessons) |
| Prophet | 3.70% | daily | Not directly comparable to hourly models |
| LightGBM | 3.06% | hourly | Gradient-boosted trees |
| XGBoost | 3.08% | hourly | Gradient-boosted trees |
| **CatBoost** ⭐ | **2.83%** | hourly | **Best model — production choice** |
| _TSO day-ahead (benchmark)_ | _4.40%_ | _hourly_ | _Official forecast, same hold-out_ |

**Baseline → winner:** Seasonal-Naive 6.44% → CatBoost 2.83% (**56% error reduction**).
**Winner vs benchmark:** CatBoost beats the TSO's 4.40% by **~36%** on identical hours.

> **A note on fairness:** the gradient-boosted models and the Seasonal-Naive baseline are
> scored on *hourly* demand (the hard target). SARIMA and Prophet were evaluated on *daily
> totals*, which smooth away the intraday peaks that drive most forecast error so their
> numbers document the modelling progression rather than competing head-to-head. The TSO's
> full-history (2020–2026) MAPE is ~3.79%; on the identical 2026 hold-out it is 4.40%, which
> is the fair comparison used throughout.

---

## 5 · Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.x** | Core language |
| **pandas / numpy** | Data manipulation, feature engineering |
| **pyarrow** | Parquet I/O for processed data |
| **scikit-learn** | TimeSeriesSplit cross-validation, metrics |
| **statsmodels** | SARIMA implementation |
| **Prophet** | Additive seasonal forecasting (Facebook/Meta) |
| **LightGBM / XGBoost / CatBoost** | Gradient-boosted tree models |
| **MLflow** | Experiment tracking (params, metrics, artifacts) |
| **Streamlit** | Interactive dashboard |
| **Plotly** | Forecast-vs-actual visualisations |
| **holidays** | German public-holiday calendar features |
| **pytest** | Pipeline unit tests |

---

## 6 · Setup & Installation

```bash
# 1. Clone
git clone https://github.com/UtkarshRai-ds/german-electricity-demand-forecasting.git
cd german-electricity-demand-forecasting

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install (editable package + dependencies)
pip install -e .
pip install -r requirements-dev.txt   # full dev stack (training, MLflow, tests)
# OR, for the dashboard only:
pip install -r requirements.txt        # slim runtime (streamlit, pandas, numpy, plotly, pyarrow)
```

---

## 7 · How to Run

```bash
# ── Full training pipeline (from project root) ───────────────────
python -m src.features.engineer        # build leak-free features.parquet
python -m src.models.baseline          # Seasonal-Naive baseline
python -m src.models.sarima            # SARIMA
python -m src.models.prophet_model     # Prophet
python -m src.models.boosting          # LightGBM / XGBoost / CatBoost (+ MLflow)
python -m src.models.export_for_app    # export artifacts for the dashboard

# ── Launch the dashboard ─────────────────────────────────────────
streamlit run app/streamlit_app.py

# ── Run the test suite ───────────────────────────────────────────
pytest

# ── Inspect experiment tracking ──────────────────────────────────
mlflow ui            # then open http://localhost:5000
```

> **Note on MLflow:** recent MLflow versions restrict the local file store. If a run errors,
> set `MLFLOW_ALLOW_FILE_STORE=true` in your shell before running.

---

## 8 · Feature Engineering

All features are shifted by the 24-hour forecast horizon so the model only ever uses
information available at forecast time. Calendar features are known in advance and are not shifted.

| Feature | Type | Rationale |
|---|---|---|
| `lag_168` | Lag | Demand exactly one week ago - captures hour *and* day type together; the single strongest predictor |
| `lag_24` | Lag | Demand 24 hours ago — anchors the daily cycle |
| `hour` | Calendar | Hour of day — the ~20,000 MWh morning/evening swing |
| `rolling_mean_24h` | Rolling | Recent demand level (window ends ≥24h before target) |
| `day_of_week` | Calendar | Weekday vs weekend demand differs 10–20% |
| `is_holiday` | Calendar | German public holidays behave like Sundays |
| `is_weekend` | Calendar | Weekend industrial slowdown |
| `month` | Calendar | Slow seasonal cycle (winter ~10% above summer) |
| `rolling_std_24h` | Rolling | Recent volatility |
| `rolling_mean_168h` | Rolling | Weekly demand level |
| `temp_2m` | Weather | Temperature (shifted; low importance — demand is calendar-driven) |
| `temp_lag_24` | Weather | Prior-day temperature |
| `temp_rolling_mean_72h` | Weather | Smoothed temperature trend |

> **Insight:** lag and calendar features dominate; temperature contributes little. For electricity
> *demand* (unlike renewable *generation*) human routines matter far more than weather.

---

## 9 · Key Decisions & Lessons

- **Found and fixed data leakage in my own work.** A mid-project audit revealed that rolling and
  temperature features were computed over windows that included the predicted hour and silently leaking the target. Fixing it (shifting every such feature by the 24h horizon) raised CatBoost's honest MAPE from an inflated **2.24% to 2.83%**. The full before/after is in [`reports/leak_audit.md`](reports/leak_audit.md). **The honest number, with an audit trail, is worth more than the inflated one.**

- **SARIMA's divergence is a documented result, not a hidden failure.** SARIMA assumes a stable seasonal pattern; German demand went through structural shifts (COVID-2020, the 2022 energy crisis), so its forecast drifted badly (36.96% MAPE). Keeping it in the line-up and explaining *why* it failed and justifies the progression to tree-based models that make no stability assumption.

- **Benchmarks must match the test window.** The TSO's headline accuracy (~3.79%) is a full-history figure. Compared fairly on the identical 2026 hold-out it is 4.40% the number used throughout. Always benchmark on the same period you score your own model on.

- **Gradient boosting over classical methods.** All three boosting models beat the benchmark;
  CatBoost won on honest MAPE and handles the categorical-style calendar features cleanly.

- **Reproducibility first.** Every run is tracked in MLflow and the dashboard reads pre-exported artifacts, so the deployed app never retrains and the numbers can be regenerated end-to-end.

---

## 10 · Limitations & Future Work

**Limitations**
- **Weather is observed, not forecasted.** Temperature features use observed values shifted 24h as a stand-in for a real day-ahead weather forecast; true operational error would be marginally higher (though temperature has low feature importance here).
- **Single test window.** Scored on one continuous period (H1 2026), so 2.83% is one realisation, not a distribution across many windows.
- **National aggregate only.** Forecasts total German demand with no regional, generation-mix, or price breakdown.
- **Atypical regimes retained.** COVID-2020 and the 2022 energy crisis remain in the training
  data un-downweighted, which may bias the model toward those conditions.

**Future Work**
- Use archived day-ahead weather *forecasts* (Open-Meteo historical-forecast API) for true
  operational realism.
- Rolling-origin backtesting across many windows to report a confidence interval, not a single number.
- Probabilistic forecasts (prediction intervals) for grid reserve planning.
- Hyperparameter tuning (Optuna) on the boosting models, currently near-default.
- Re-run Prophet at hourly resolution for a fully like-for-like comparison.

---

## 11 · File Structure

```
german-electricity-demand-forecasting/
├── app/
│   └── streamlit_app.py          # Interactive dashboard (4 pages)
├── src/
│   ├── data/
│   │   ├── loader.py             # Load raw SMARD / Open-Meteo
│   │   ├── cleaner.py            # Clean & align to hourly index
│   │   ├── quality.py            # Data-quality checks
│   │   └── weather.py            # Weather ingestion
│   ├── features/
│   │   └── engineer.py           # Leak-free feature engineering
│   └── models/
│       ├── baseline.py           # Seasonal-Naive
│       ├── sarima.py             # SARIMA
│       ├── prophet_model.py      # Prophet
│       ├── boosting.py           # LightGBM / XGBoost / CatBoost
│       └── export_for_app.py     # Export dashboard artifacts
├── tests/
│   └── test_pipeline.py          # pytest suite (8 tests)
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
├── reports/
│   └── leak_audit.md             # Leakage before/after audit
├── data/
│   ├── raw/                      # Raw SMARD CSVs
│   ├── processed/                # Cleaned parquet (features.parquet)
│   └── app/                      # Pre-exported dashboard artifacts
├── models/                       # Trained .pkl + result JSONs + plots
├── requirements.txt              # Slim runtime deps (dashboard)
├── requirements-dev.txt          # Full dev deps (training, MLflow, tests)
├── setup.py
└── README.md
```

---

## 📚 Related Work

This project's modelling choices follow recent peer-reviewed literature on gradient-boosted electricity load forecasting:

- Muqtadir, A., Li, B., Ying, Z., Songsong, C., & Kazmi, S. N. (2025). *"Nowcasting the next
  hour of residential load using boosting ensemble machines."* Scientific Reports, 15, 7157.
  https://www.nature.com/articles/s41598-025-91767-6 — Integrates LightGBM, XGBoost, and CatBoost
  for short-term load forecasting, motivating this project's three-model boosting stack.

- Song, K.-M., Kim, T.-G., Cho, S.-M., & Song, K.-B. (2025). *"XGBoost-Based Very Short-Term
  Load Forecasting Using Day-Ahead Load Forecasting Results."* Electronics, 14(18), 3747.
  DOI: 10.3390/electronics14183747 — Incorporates the TSO day-ahead forecast as a model input,
  directly relevant to this project's TSO benchmark comparison.

---

## 📄 Data License & Attribution

Electricity load data from [SMARD.de](https://www.smard.de) (Bundesnetzagentur), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Weather data from
[Open-Meteo](https://open-meteo.com), also CC BY 4.0. Attribution to these sources is required
under the licence terms and is provided here accordingly.

---

## 🔗 Author

**Utkarsh Rai** — Data Scientist (Berlin) · [GitHub](https://github.com/UtkarshRai-ds)
