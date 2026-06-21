from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_MODELS_DIR    = _PROJECT_ROOT / "models"

_TRAIN_END  = pd.Timestamp("2026-01-01", tz="UTC")
_LAG        = 168  # hours — same hour last week

# TSO benchmark published in EDA notebook
_TSO_MAE  = 2048.6
_TSO_MAPE = 3.80


class SeasonalNaiveBaseline:
    def __init__(self, features: pd.DataFrame) -> None:
        self.features = features
        self.train: pd.DataFrame | None = None
        self.test:  pd.DataFrame | None = None
        self.predictions: pd.Series | None = None
        self.results: dict | None = None

    # ── split ──────────────────────────────────────────────────────────────────

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.train = self.features[self.features.index < _TRAIN_END]
        self.test  = self.features[self.features.index >= _TRAIN_END]
        return self.train, self.test

    # ── predict ────────────────────────────────────────────────────────────────

    def predict(self) -> pd.Series:
        if self.test is None:
            self.split()

        # lag_168 is already in the feature set — use it directly when available,
        # but fall back to a manual lookup so the class works on a bare load+temp df too.
        if "lag_168" in self.test.columns:
            preds = self.test["lag_168"].copy()
        else:
            all_load = self.features["actual_load"]
            preds = self.test.index.map(
                lambda ts: all_load.get(ts - pd.Timedelta(hours=_LAG))
            )
            preds = pd.Series(preds, index=self.test.index, name="predicted")

        self.predictions = preds
        return self.predictions

    # ── evaluate ───────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        if self.predictions is None:
            self.predict()

        actual = self.test["actual_load"]
        pred   = self.predictions

        # align on common non-null index
        mask   = actual.notna() & pred.notna()
        actual = actual[mask]
        pred   = pred[mask]

        mae  = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)

        self.results = {
            "mae":  round(mae,  2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 4),
            "n_test_rows": int(mask.sum()),
            "test_start":  str(self.test.index.min()),
            "test_end":    str(self.test.index.max()),
        }
        return self.results

    # ── plot ───────────────────────────────────────────────────────────────────

    def plot(self) -> Path:
        if self.results is None:
            self.evaluate()

        two_weeks = self.test.index[:24 * 14]
        actual    = self.test.loc[two_weeks, "actual_load"]
        pred      = self.predictions.loc[two_weeks]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(actual.index, actual.values, label="Actual",    linewidth=1.2)
        ax.plot(pred.index,   pred.values,   label="Predicted (lag-168h)",
                linewidth=1.0, linestyle="--", alpha=0.85)

        ax.set_title("Seasonal Naïve Baseline — first 2 weeks of test set (2026)")
        ax.set_xlabel("Timestamp (UTC)")
        ax.set_ylabel("Grid load (MWh)")
        ax.legend()
        fig.tight_layout()

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = _MODELS_DIR / "baseline_forecast.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ── save results ───────────────────────────────────────────────────────────

    def save_results(self) -> Path:
        if self.results is None:
            self.evaluate()

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = _MODELS_DIR / "baseline_results.json"
        out.write_text(json.dumps(self.results, indent=2))
        return out

    # ── summary table ──────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        if self.results is None:
            self.evaluate()

        r = self.results
        header = f"{'Metric':<8}  {'Baseline':>12}  {'TSO benchmark':>14}"
        sep    = "─" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep)
        print(f"{'MAE':<8}  {r['mae']:>11.2f}  {_TSO_MAE:>13.1f}  MWh")
        print(f"{'RMSE':<8}  {r['rmse']:>11.2f}  {'—':>13}")
        print(f"{'MAPE':<8}  {r['mape']:>10.4f}%  {_TSO_MAPE:>12.2f}%")
        print(sep)
        print(f"  test rows : {r['n_test_rows']:,}")
        print(f"  test span : {r['test_start']}  →  {r['test_end']}")
        print(f"{sep}\n")


if __name__ == "__main__":
    features = pd.read_parquet(_PROCESSED_DIR / "features.parquet")

    model = SeasonalNaiveBaseline(features)
    model.split()
    model.predict()
    model.evaluate()
    model.print_summary()

    png  = model.plot()
    json_path = model.save_results()
    print(f"Plot   → {png}")
    print(f"Results → {json_path}")
