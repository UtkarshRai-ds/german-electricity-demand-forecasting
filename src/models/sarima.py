from __future__ import annotations

# WHY daily aggregation instead of raw hourly data
# ─────────────────────────────────────────────────
# Electricity load has three simultaneous seasonal cycles:
#   • intraday  : period = 24   (peak morning / evening, trough overnight)
#   • intraweek : period = 168  (weekday vs. weekend shape)
#   • intrayear : period = 8 766 (summer vs. winter demand)
#
# Plain ARIMA has no seasonal term at all, so it cannot model any of these.
# SARIMA adds one seasonal block (P,D,Q,s), which handles exactly one period.
# Fitting SARIMA on hourly data with s=168 (weekly) would require estimating
# 168 seasonal-difference lags and is numerically unstable on most solvers.
#
# Daily aggregation collapses the intraday cycle: summing 24 hourly readings
# into one daily total removes the period-24 pattern entirely.  The weekly
# cycle now has period = 7 days, which SARIMA(p,d,q)(P,D,Q,7) models cleanly
# with stable maximum-likelihood estimation.
#
# The intrayear cycle remains, but with ~6 years of daily data the seasonal
# difference (D=1 at s=7) and the trend difference (d=1) together absorb
# the slow annual drift without requiring a third seasonal block.
#
# Honest interview answer: "ARIMA cannot handle seasonality at all.
# SARIMA handles one season. Daily aggregation lets us pick the most
# actionable season (weekly) while keeping the model tractable."

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_MODELS_DIR    = _PROJECT_ROOT / "models"

_TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")
_TSO_MAPE  = 3.80

_ORDER    = (2, 1, 2)
_SEASONAL = (1, 0, 1, 7)


class SARIMAModel:
    def __init__(self, features: pd.DataFrame) -> None:
        self.features = features
        self.daily:       pd.Series | None = None
        self.train_daily: pd.Series | None = None
        self.test_daily:  pd.Series | None = None
        self.forecast:    pd.Series | None = None
        self.results:     dict       | None = None

    # ── aggregate ──────────────────────────────────────────────────────────────

    def aggregate(self) -> pd.Series:
        # tz_localize guard: features index must be UTC-aware for correct resampling
        idx = self.features.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
            self.features = self.features.copy()
            self.features.index = idx

        self.daily = (
            self.features["actual_load"]
            .resample("D")
            .sum()
            .rename("actual_load_daily")
        )
        return self.daily

    # ── split ──────────────────────────────────────────────────────────────────

    def split(self) -> tuple[pd.Series, pd.Series]:
        if self.daily is None:
            self.aggregate()

        self.train_daily = self.daily[self.daily.index < _TRAIN_END]
        self.test_daily  = self.daily[self.daily.index >= _TRAIN_END]
        return self.train_daily, self.test_daily

    # ── fit & forecast ─────────────────────────────────────────────────────────

    def fit_forecast(self) -> pd.Series:
        if self.train_daily is None:
            self.split()

        print(
            f"Fitting SARIMA{_ORDER}{_SEASONAL} on "
            f"{len(self.train_daily):,} daily observations …"
        )
        model = SARIMAX(
            self.train_daily,
            order=_ORDER,
            seasonal_order=_SEASONAL,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)

        self.forecast = fit.forecast(steps=len(self.test_daily))
        self.forecast.index = self.test_daily.index
        self.forecast.name  = "forecast"
        return self.forecast

    # ── evaluate ───────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        if self.forecast is None:
            self.fit_forecast()

        actual = self.test_daily
        pred   = self.forecast

        mask   = actual.notna() & pred.notna()
        actual = actual[mask]
        pred   = pred[mask]

        mae  = float(np.mean(np.abs(actual - pred)))
        rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
        mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)

        self.results = {
            "mae":         round(mae,  2),
            "rmse":        round(rmse, 2),
            "mape":        round(mape, 4),
            "n_test_days": int(mask.sum()),
            "test_start":  str(self.test_daily.index.min().date()),
            "test_end":    str(self.test_daily.index.max().date()),
        }
        return self.results

    # ── plot ───────────────────────────────────────────────────────────────────

    def plot(self) -> Path:
        if self.results is None:
            self.evaluate()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            self.test_daily.index, self.test_daily.values,
            label="Actual (daily total)", linewidth=1.2,
        )
        ax.plot(
            self.forecast.index, self.forecast.values,
            label=f"SARIMA{_ORDER}{_SEASONAL} forecast",
            linewidth=1.0, linestyle="--", alpha=0.85,
        )
        ax.set_title("SARIMA Forecast — test set 2026 (daily totals)")
        ax.set_xlabel("Date (UTC)")
        ax.set_ylabel("Grid load (MWh / day)")
        ax.legend()
        fig.tight_layout()

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = _MODELS_DIR / "sarima_forecast.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ── save results ───────────────────────────────────────────────────────────

    def save_results(self) -> Path:
        if self.results is None:
            self.evaluate()

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = _MODELS_DIR / "sarima_results.json"
        out.write_text(json.dumps(self.results, indent=2))
        return out

    # ── summary table ──────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        if self.results is None:
            self.evaluate()

        r = self.results

        if r["mape"] > 20.0:
            print("WARNING: Model diverged — check differencing order")
            print(f"  MAPE = {r['mape']:.4f}%  |  test days: {r['n_test_days']:,}")
            print(f"  test span: {r['test_start']}  →  {r['test_end']}\n")
            return

        note = "(daily — not directly comparable to hourly TSO benchmark)"
        header = f"{'Metric':<8}  {'SARIMA':>12}  {'TSO benchmark':>14}"
        sep    = "─" * max(len(header), len(note) + 2)
        print(f"\n{sep}")
        print(header)
        print(sep)
        print(f"{'MAE':<8}  {r['mae']:>11.2f}  {'—':>13}  MWh/day")
        print(f"{'RMSE':<8}  {r['rmse']:>11.2f}  {'—':>13}  MWh/day")
        print(f"{'MAPE':<8}  {r['mape']:>10.4f}%  {_TSO_MAPE:>12.2f}%  {note}")
        print(sep)
        print(f"  test days : {r['n_test_days']:,}")
        print(f"  test span : {r['test_start']}  →  {r['test_end']}")
        print(f"{sep}\n")


if __name__ == "__main__":
    features = pd.read_parquet(_PROCESSED_DIR / "features.parquet")

    model = SARIMAModel(features)
    model.aggregate()
    model.split()
    model.fit_forecast()
    model.evaluate()
    model.print_summary()

    png       = model.plot()
    json_path = model.save_results()
    print(f"Plot    → {png}")
    print(f"Results → {json_path}")
