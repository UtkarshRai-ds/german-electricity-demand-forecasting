"""
export_for_app.py
─────────────────
Generates the data artifacts the Streamlit dashboard reads, so every number the
app shows is the honest, post-leak-fix result — computed once here, not in the app.

RUN ORDER (from project root):
    python -m src.features.engineer        # regenerate leak-free features.parquet
    python -m src.models.boosting          # overwrite the 3 model pickles (honest)
    python -m src.models.export_for_app    # <-- this script

This script LOADS the freshly-trained pickles (it does NOT retrain — boosting.py
is the single source of training truth). The CatBoost pickle generates the
production forecast line; the TSO benchmark is derived IN-CODE from
forecast_load.parquet on the exact 2026 holdout, so nothing is hardcoded.

Outputs (data/app/):
    forecast_vs_actual.csv   timestamp, actual, catboost, tso   (2026 holdout, hourly)
    model_results.json       per-model MAPE, benchmark (same-holdout + full-history), leak audit
    feature_importance.csv   feature, importance  (CatBoost)
    eda_profiles.json        hour-of-day / day-of-week / month average load profiles
    dataset_summary.json     row counts, date span, headline KPIs
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "data" / "processed"
_MODELS = _ROOT / "models"
_APP_DIR = _ROOT / "data" / "app"
_APP_DIR.mkdir(parents=True, exist_ok=True)

# ── Config (kept in sync with boosting.py) ────────────────────────────────────
TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "lag_24", "lag_168",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h",
    "temp_2m", "temp_lag_24", "temp_rolling_mean_72h",
]
# TSO full-history MAPE as documented in the EDA notebook (2020–2026).
# Shown alongside the same-holdout figure for transparency; NOT the comparison number.
TSO_FULL_HISTORY_MAPE = 3.79
# Pre-fix (leaky) MAPEs — kept only to display the honest before/after.
LEAKED_MAPE = {"LightGBM": 2.7686, "XGBoost": 2.5829, "CatBoost": 2.2419}


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    m = (actual != 0) & ~np.isnan(pred)
    return float(np.mean(np.abs((actual[m] - pred[m]) / actual[m])) * 100)


def main() -> None:
    # ── Load leak-free features ───────────────────────────────────────────────
    df = pd.read_parquet(_PROCESSED / "features.parquet")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    test = df[df.index >= TRAIN_END]
    X_te = test[FEATURES].astype(float).values
    y_te = test["actual_load"].values

    # ── Load fresh pickles (trained by boosting.py on the leak-free parquet) ───
    models = {}
    for name in ("lightgbm", "xgboost", "catboost"):
        pkl = _MODELS / f"{name}_model.pkl"
        if not pkl.exists():
            raise FileNotFoundError(
                f"{pkl} missing. Run `python -m src.models.boosting` first so the "
                f"dashboard loads honest, post-leak-fix models."
            )
        models[name] = joblib.load(pkl)

    honest_mape = {
        "LightGBM": round(mape(y_te, models["lightgbm"].predict(X_te)), 4),
        "XGBoost": round(mape(y_te, models["xgboost"].predict(X_te)), 4),
        "CatBoost": round(mape(y_te, models["catboost"].predict(X_te)), 4),
    }
    catboost_pred = models["catboost"].predict(X_te)

    # ── TSO benchmark: derive IN-CODE on the SAME 2026 holdout ────────────────
    tso = pd.read_parquet(_PROCESSED / "forecast_load.parquet")
    if tso.index.tz is None:
        tso.index = tso.index.tz_localize("UTC")  # forecast_load is tz-naive
    tso_aligned = test[["actual_load"]].join(tso, how="left")["forecast_load"]
    tso_holdout_mape = round(mape(y_te, tso_aligned.values), 4)

    # ── 1. Forecast vs actual vs TSO (three-line overlay source) ──────────────
    pd.DataFrame({
        "timestamp": test.index,
        "actual": y_te,
        "catboost": catboost_pred,
        "tso": tso_aligned.values,
    }).to_csv(_APP_DIR / "forecast_vs_actual.csv", index=False)

    # ── 2. Model results JSON ─────────────────────────────────────────────────
    best_mape = honest_mape["CatBoost"]
    results = {
        "benchmark": {
            "same_holdout_mape": tso_holdout_mape,        # the comparison number
            "full_history_mape": TSO_FULL_HISTORY_MAPE,   # context only
            "source": "ENTSO-E / German TSO composite day-ahead forecast",
        },
        "best_model": "CatBoost",
        "best_mape": best_mape,
        "improvement_vs_benchmark_pct": round((1 - best_mape / tso_holdout_mape) * 100, 1),
        "models": [
            {"name": "Seasonal Naive", "mape": 6.44, "resolution": "hourly", "note": "simple baseline"},
            {"name": "SARIMA", "mape": 36.96, "resolution": "daily", "note": "diverged — structural breaks"},
            {"name": "Prophet", "mape": 3.70, "resolution": "daily", "note": "not directly comparable to hourly"},
            {"name": "LightGBM", "mape": honest_mape["LightGBM"], "resolution": "hourly", "note": "gradient-boosted trees"},
            {"name": "XGBoost", "mape": honest_mape["XGBoost"], "resolution": "hourly", "note": "gradient-boosted trees"},
            {"name": "CatBoost", "mape": honest_mape["CatBoost"], "resolution": "hourly", "note": "best"},
        ],
        "leak_audit": [
            {"model": k, "before": LEAKED_MAPE[k], "after": honest_mape[k],
             "delta_pp": round(honest_mape[k] - LEAKED_MAPE[k], 2)}
            for k in ("LightGBM", "XGBoost", "CatBoost")
        ],
    }
    (_APP_DIR / "model_results.json").write_text(json.dumps(results, indent=2))

    # ── 3. Feature importance (CatBoost) ──────────────────────────────────────
    cb = models["catboost"]
    importances = (cb.get_feature_importance()
                   if hasattr(cb, "get_feature_importance")
                   else cb.feature_importances_)
    pd.DataFrame({"feature": FEATURES, "importance": importances}) \
        .sort_values("importance", ascending=False) \
        .to_csv(_APP_DIR / "feature_importance.csv", index=False)

    # ── 4. EDA seasonality profiles (from actual_load, full series) ───────────
    load = df["actual_load"]
    berlin = df.index.tz_convert("Europe/Berlin")
    profiles = {
        "hour_of_day": load.groupby(berlin.hour).mean().round(1).to_dict(),
        "day_of_week": load.groupby(berlin.dayofweek).mean().round(1).to_dict(),
        "month": load.groupby(berlin.month).mean().round(1).to_dict(),
    }
    # JSON keys must be strings
    profiles = {k: {str(kk): vv for kk, vv in v.items()} for k, v in profiles.items()}
    (_APP_DIR / "eda_profiles.json").write_text(json.dumps(profiles, indent=2))

    # ── 5. Dataset summary ────────────────────────────────────────────────────
    summary = {
        "n_rows": int(len(df)),
        "n_features": len(FEATURES),
        "date_start": str(df.index.min().date()),
        "date_end": str(df.index.max().date()),
        "test_start": str(test.index.min().date()),
        "test_end": str(test.index.max().date()),
        "n_test_hours": int(len(test)),
    }
    (_APP_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    # ── Console summary ───────────────────────────────────────────────────────
    print("Exported app artifacts →", _APP_DIR)
    for name, val in honest_mape.items():
        print(f"  {name:<10} honest MAPE: {val:.4f}%")
    print(f"  TSO (same 2026 holdout)  : {tso_holdout_mape:.4f}%")
    print(f"  TSO (full history, notebook): {TSO_FULL_HISTORY_MAPE:.2f}%")
    print(f"  CatBoost beats same-holdout TSO by "
          f"{results['improvement_vs_benchmark_pct']:.1f}%")


if __name__ == "__main__":
    main()
