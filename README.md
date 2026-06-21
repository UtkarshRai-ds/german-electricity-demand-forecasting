# German Electricity Demand Forecasting

Short-term electricity demand forecasting for the German grid using
statistical and gradient-boosting models with MLflow experiment tracking
and a Streamlit dashboard.

## Project structure

```
├── src/
│   ├── data/        # Data loading, validation, preprocessing
│   ├── features/    # Lag, rolling, calendar, and weather features
│   └── models/      # Baseline, SARIMA, XGBoost/LightGBM/CatBoost + backtesting
├── app/             # Streamlit dashboard
├── tests/           # Unit tests (pytest)
├── notebooks/       # Exploratory data analysis
├── data/
│   ├── raw/         # Raw source files (git-ignored)
│   └── processed/   # Cleaned & feature-enriched parquet files (git-ignored)
├── models/          # Serialised model artefacts (git-ignored)
└── mlruns/          # MLflow experiment tracking (git-ignored)
```

## Quick start

```bash
pip install -e .
pip install -r requirements.txt
pytest
streamlit run app/dashboard.py
```
