from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_MODELS_DIR    = _PROJECT_ROOT / "models"

_TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")
def _tso_benchmark_mape() -> float:
    feats = pd.read_parquet(_PROCESSED_DIR / "features.parquet")
    if feats.index.tz is None:
        feats.index = feats.index.tz_localize("UTC")
    test = feats[feats.index >= _TRAIN_END]
    tso = pd.read_parquet(_PROCESSED_DIR / "forecast_load.parquet")
    if tso.index.tz is None:
        tso.index = tso.index.tz_localize("UTC")
    aligned = test[["actual_load"]].join(tso, how="left")
    a = aligned["actual_load"].values
    p = aligned["forecast_load"].values
    m = (a != 0) & ~np.isnan(p)
    return round(float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100), 4)


_TSO_MAPE = _tso_benchmark_mape()

_FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "lag_24", "lag_168",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h",
    "temp_2m", "temp_lag_24", "temp_rolling_mean_72h",
]
_TARGET = "actual_load"


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def _mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


class BoostingForecaster:
    def __init__(self, features: pd.DataFrame) -> None:
        self.features = features
        self.X_train: pd.DataFrame | None = None
        self.X_test:  pd.DataFrame | None = None
        self.y_train: pd.Series    | None = None
        self.y_test:  pd.Series    | None = None
        self.model_results: list[dict]    = []

    # ── split ──────────────────────────────────────────────────────────────────

    def split(self) -> None:
        idx = self.features.index
        if idx.tz is None:
            self.features = self.features.copy()
            self.features.index = idx.tz_localize("UTC")

        train = self.features[self.features.index < _TRAIN_END]
        test  = self.features[self.features.index >= _TRAIN_END]

        self.X_train = train[_FEATURE_COLS].astype(float)
        self.y_train = train[_TARGET]
        self.X_test  = test[_FEATURE_COLS].astype(float)
        self.y_test  = test[_TARGET]

    # ── cross-validate ─────────────────────────────────────────────────────────

    def _cv_mape(self, estimator) -> float:
        tscv   = TimeSeriesSplit(n_splits=5)
        X, y   = self.X_train.values, self.y_train.values
        scores = []
        for train_idx, val_idx in tscv.split(X):
            estimator.fit(X[train_idx], y[train_idx])
            pred = estimator.predict(X[val_idx])
            scores.append(_mape(y[val_idx], pred))
        return float(np.mean(scores))

    # ── train all models ───────────────────────────────────────────────────────

    def train(self) -> list[dict]:
        if self.X_train is None:
            self.split()

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)

        candidates = [
            ("LightGBM",  LGBMRegressor(verbose=-1)),
            ("XGBoost",   XGBRegressor(verbosity=0)),
            ("CatBoost",  CatBoostRegressor(verbose=0)),
        ]

        mlflow.set_tracking_uri(f"file:///{_PROJECT_ROOT / 'mlruns'}")
        mlflow.set_experiment("boosting_forecasters")

        for name, estimator in candidates:
            print(f"[{name}] cross-validating …")
            cv_mape = self._cv_mape(estimator)

            print(f"[{name}] fitting on full train set …")
            estimator.fit(self.X_train.values, self.y_train.values)

            pred = estimator.predict(self.X_test.values)
            actual = self.y_test.values

            test_mae  = _mae(actual, pred)
            test_rmse = _rmse(actual, pred)
            test_mape = _mape(actual, pred)

            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_metric("cv_mape",   round(cv_mape,   4))
                mlflow.log_metric("test_mape", round(test_mape, 4))
                mlflow.log_metric("test_mae",  round(test_mae,  2))
                mlflow.log_metric("test_rmse", round(test_rmse, 2))

            pkl_path = _MODELS_DIR / f"{name.lower()}_model.pkl"
            joblib.dump(estimator, pkl_path)

            self.model_results.append({
                "name":      name,
                "estimator": estimator,
                "cv_mape":   round(cv_mape,   4),
                "test_mae":  round(test_mae,  2),
                "test_rmse": round(test_rmse, 2),
                "test_mape": round(test_mape, 4),
                "pkl":       str(pkl_path),
            })
            print(
                f"[{name}] CV MAPE={cv_mape:.4f}%  "
                f"test MAPE={test_mape:.4f}%  saved -> {pkl_path.name}"
            )

        return self.model_results

    # ── feature importance plot ────────────────────────────────────────────────

    def plot_feature_importance(self) -> Path:
        if not self.model_results:
            self.train()

        best = min(self.model_results, key=lambda r: r["test_mape"])
        estimator = best["estimator"]
        name      = best["name"]

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        else:
            raise AttributeError(f"{name} has no feature_importances_ attribute")

        order = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [_FEATURE_COLS[i] for i in order],
            importances[order],
        )
        ax.set_title(f"Feature importance — {name} (best test MAPE)")
        ax.set_xlabel("Importance")
        fig.tight_layout()

        out = _MODELS_DIR / "feature_importance.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ── save best results ──────────────────────────────────────────────────────

    def save_results(self) -> Path:
        if not self.model_results:
            self.train()

        best = min(self.model_results, key=lambda r: r["test_mape"])
        out  = _MODELS_DIR / "boosting_results.json"
        out.write_text(json.dumps(
            {"best_model": best["name"], "best_test_mape": best["test_mape"]},
            indent=2,
        ))
        return out

    # ── comparison table ───────────────────────────────────────────────────────

    def print_summary(self) -> None:
        if not self.model_results:
            self.train()

        header = f"{'Model':<12}  {'CV MAPE':>10}  {'Test MAPE':>10}  {'TSO':>8}  {'Test MAE':>10}  {'Test RMSE':>11}"
        sep    = "-" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)
        for r in sorted(self.model_results, key=lambda x: x["test_mape"]):
            print(
                f"{r['name']:<12}  {r['cv_mape']:>9.4f}%  "
                f"{r['test_mape']:>9.4f}%  {_TSO_MAPE:>7.2f}%  "
                f"{r['test_mae']:>10.2f}  {r['test_rmse']:>11.2f}"
            )
        print(sep)
        best = min(self.model_results, key=lambda r: r["test_mape"])
        print(f"  best model : {best['name']}  (test MAPE {best['test_mape']:.4f}%)\n")


if __name__ == "__main__":
    features = pd.read_parquet(_PROCESSED_DIR / "features.parquet")

    forecaster = BoostingForecaster(features)
    forecaster.split()
    forecaster.train()
    forecaster.print_summary()

    png       = forecaster.plot_feature_importance()
    json_path = forecaster.save_results()
    print(f"Feature importance -> {png}")
    print(f"Results            -> {json_path}")
