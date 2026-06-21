from __future__ import annotations

# WHY Prophet over SARIMA for German electricity load
# ─────────────────────────────────────────────────────
# ARIMA-family models assume a single data-generating process that is stationary
# after differencing.  German electricity demand violates this in two episodes:
#
#   2020 Q1-Q2  COVID-19 lockdowns caused an abrupt, sustained demand drop of
#               ~5-8 % that reversed gradually over 18 months.  This is a
#               structural break, not a transient shock, so differencing cannot
#               remove it — it creates a spike in the differenced series that
#               propagates through the AR and MA terms.
#
#   2022 H2     The energy crisis (Russian gas supply cuts) drove efficiency
#               programmes and industrial curtailment.  Demand fell ~10 % vs.
#               the pre-crisis trend and has not fully recovered.  Again a
#               level-shift that violates the stationarity assumption.
#
# Prophet's piecewise linear trend models these as changepoints: it places
# potential breakpoints at regular intervals across the training period and
# uses a sparse regularisation prior (changepoint_prior_scale) to select only
# those where the trend genuinely changes direction.  The 2020 and 2022 breaks
# are detected automatically without the user needing to hard-code their dates.
#
# Secondary advantages:
#   • Additive yearly + weekly seasonality handles both cycles simultaneously,
#     unlike SARIMA which is limited to one seasonal period per model.
#   • German public holidays are injected as a regressor with a single call,
#     rather than requiring manual indicator columns.
#   • Missing days and irregular calendars (Easter, Whit Monday) are handled
#     transparently through the additive decomposition.
#
# Honest interview answer: "ARIMA stationarity breaks down whenever the data
# contains structural level shifts.  Prophet's changepoint mechanism was
# designed for exactly this pattern and lets the trend adapt while keeping
# seasonality stable."

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from prophet import Prophet

# silence Prophet's verbose Stan compiler output
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_MODELS_DIR    = _PROJECT_ROOT / "models"

_TRAIN_END       = pd.Timestamp("2026-01-01", tz="UTC")
_TSO_MAPE        = 3.80
_NAIVE_MAPE      = 6.44


class ProphetForecaster:
    def __init__(self, features: pd.DataFrame) -> None:
        self.features  = features
        self.daily:       pd.Series    | None = None
        self.train_prophet: pd.DataFrame | None = None  # ds / y format
        self.test_prophet:  pd.DataFrame | None = None
        self.test_daily:  pd.Series    | None = None
        self.forecast_df: pd.DataFrame | None = None
        self.model:       Prophet      | None = None
        self.results:     dict         | None = None

    # ── aggregate ──────────────────────────────────────────────────────────────

    def aggregate(self) -> pd.Series:
        idx = self.features.index
        if idx.tz is None:
            self.features = self.features.copy()
            self.features.index = idx.tz_localize("UTC")

        self.daily = (
            self.features["actual_load"]
            .resample("D")
            .sum()
            .rename("actual_load_daily")
        )
        return self.daily

    # ── split & format ─────────────────────────────────────────────────────────

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.daily is None:
            self.aggregate()

        train_raw = self.daily[self.daily.index < _TRAIN_END]
        test_raw  = self.daily[self.daily.index >= _TRAIN_END]

        self.test_daily = test_raw

        # Prophet requires tz-naive ds column
        def _to_prophet(s: pd.Series) -> pd.DataFrame:
            ds = s.index.tz_localize(None) if s.index.tz is not None else s.index
            return pd.DataFrame({"ds": ds, "y": s.values})

        self.train_prophet = _to_prophet(train_raw)
        self.test_prophet  = _to_prophet(test_raw)
        return self.train_prophet, self.test_prophet

    # ── fit & forecast ─────────────────────────────────────────────────────────

    def fit_forecast(self) -> pd.DataFrame:
        if self.train_prophet is None:
            self.split()

        print(
            f"Fitting Prophet on {len(self.train_prophet):,} daily observations …"
        )
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_mode="additive",
        )
        self.model.add_country_holidays(country_name="DE")
        self.model.fit(self.train_prophet)

        self.forecast_df = self.model.predict(self.test_prophet)
        return self.forecast_df

    # ── evaluate ───────────────────────────────────────────────────────────────

    def evaluate(self) -> dict:
        if self.forecast_df is None:
            self.fit_forecast()

        actual = self.test_daily.values
        pred   = self.forecast_df["yhat"].values

        mask   = np.isfinite(actual) & np.isfinite(pred)
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

        # actual
        ax.plot(
            self.test_daily.index.tz_localize(None),
            self.test_daily.values,
            label="Actual (daily total)", linewidth=1.2,
        )

        # forecast + uncertainty band
        fc = self.forecast_df
        ax.plot(
            fc["ds"], fc["yhat"],
            label="Prophet forecast", linewidth=1.0, linestyle="--", alpha=0.85,
        )
        ax.fill_between(
            fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
            alpha=0.15, label="80 % interval",
        )

        # changepoints that fall inside the test window (usually none, but guard anyway)
        if self.model is not None:
            cp_dates = pd.to_datetime(self.model.changepoints)
            test_start = fc["ds"].iloc[0]
            visible_cp = cp_dates[cp_dates >= test_start]
            for cp in visible_cp:
                ax.axvline(cp, color="red", linewidth=0.8, linestyle=":", alpha=0.7)

        # proxy legend handle for training changepoints (no NaT axvline needed)
        handles, labels = ax.get_legend_handles_labels()
        cp_dates_train = pd.to_datetime(self.model.changepoints) if self.model else []
        if len(cp_dates_train):
            proxy = Line2D(
                [0], [0],
                color="crimson", linewidth=0.8, linestyle="--",
                label=f"{len(cp_dates_train)} changepoints (train)",
            )
            handles.append(proxy)

        ax.set_title("Prophet Forecast — test set 2026 (daily totals)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Grid load (MWh / day)")
        ax.legend(handles=handles)
        fig.tight_layout()

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = _MODELS_DIR / "prophet_forecast.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    # ── save results ───────────────────────────────────────────────────────────

    def save_results(self) -> Path:
        if self.results is None:
            self.evaluate()

        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = _MODELS_DIR / "prophet_results.json"
        out.write_text(json.dumps(self.results, indent=2))
        return out

    # ── summary table ──────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        if self.results is None:
            self.evaluate()

        r    = self.results
        note = "(daily — not directly comparable to hourly TSO benchmark)"
        header = (
            f"{'Metric':<8}  {'Prophet':>10}  {'Seasonal Naive':>15}  {'TSO':>8}"
        )
        sep = "─" * max(len(header), len(note) + 2)
        print(f"\n{sep}")
        print(header)
        print(sep)
        print(f"{'MAE':<8}  {r['mae']:>10.2f}  {'—':>15}  {'—':>8}  MWh/day")
        print(f"{'RMSE':<8}  {r['rmse']:>10.2f}  {'—':>15}  {'—':>8}  MWh/day")
        print(
            f"{'MAPE':<8}  {r['mape']:>9.4f}%  {_NAIVE_MAPE:>14.2f}%  "
            f"{_TSO_MAPE:>7.2f}%  {note}"
        )
        print(sep)
        print(f"  test days : {r['n_test_days']:,}")
        print(f"  test span : {r['test_start']}  →  {r['test_end']}")
        print(f"{sep}\n")


if __name__ == "__main__":
    features = pd.read_parquet(_PROCESSED_DIR / "features.parquet")

    model = ProphetForecaster(features)
    model.aggregate()
    model.split()
    model.fit_forecast()
    model.evaluate()
    model.print_summary()

    png       = model.plot()
    json_path = model.save_results()
    print(f"Plot    → {png}")
    print(f"Results → {json_path}")
