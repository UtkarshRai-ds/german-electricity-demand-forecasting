from __future__ import annotations

from pathlib import Path

import holidays
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

_DE_HOLIDAYS = holidays.country_holidays("DE", years=range(2020, 2027))


def build_features(
    actual_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    horizon: int = 24,
) -> pd.DataFrame:
    """Join load + temperature, engineer calendar/lag/rolling features, return UTC DataFrame.

    All features derived from actual_load or temperature are shifted by `horizon` hours
    so that row H only contains information available at H - horizon (no target leakage).
    Calendar features (hour, day_of_week, month, is_weekend, is_holiday) are not shifted
    because they are known in advance for any forecast horizon.
    """
    # ── 1. Inner join on UTC DatetimeIndex ────────────────────────────────────
    if actual_df.index.tz is None:
        actual_df = actual_df.copy()
        actual_df.index = actual_df.index.tz_localize("UTC")
    if temp_df.index.tz is None:
        temp_df = temp_df.copy()
        temp_df.index = temp_df.index.tz_localize("UTC")

    df = actual_df[["actual_load"]].join(temp_df[["temperature_2m"]], how="inner")

    # ── 2. Berlin wall-clock index for calendar features only ─────────────────
    berlin = df.index.tz_convert("Europe/Berlin")

    df["hour"]        = berlin.hour
    df["day_of_week"] = berlin.dayofweek
    df["month"]       = berlin.month
    df["is_weekend"]  = df["day_of_week"] >= 5
    df["is_holiday"]  = berlin.normalize().tz_localize(None).isin(_DE_HOLIDAYS)

    # ── 3. Lag features (computed on the UTC-ordered index) ───────────────────
    load = df["actual_load"]
    df["lag_24"]  = load.shift(max(24, horizon))
    df["lag_168"] = load.shift(max(168, horizon))

    # ── 4. Rolling features ───────────────────────────────────────────────────
    # Shift by horizon BEFORE rolling so the window at H covers [H-horizon-w, H-horizon],
    # not [H-w+1, H] which would include the target hour itself.
    load_shifted = load.shift(horizon)
    df["rolling_mean_24h"]  = load_shifted.rolling(24,  min_periods=24).mean()
    df["rolling_std_24h"]   = load_shifted.rolling(24,  min_periods=24).std()
    df["rolling_mean_168h"] = load_shifted.rolling(168, min_periods=168).mean()

    # ── 5. Temperature features ───────────────────────────────────────────────
    temp = df.pop("temperature_2m")
    temp_shifted = temp.shift(horizon)
    df["temp_2m"]               = temp_shifted
    df["temp_lag_24"]           = temp.shift(max(24, horizon))
    df["temp_rolling_mean_72h"] = temp_shifted.rolling(72, min_periods=72).mean()

    # ── 6. Drop lag-warmup rows ───────────────────────────────────────────────
    df = df.dropna()

    # ── 7. Persist ────────────────────────────────────────────────────────────
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PROCESSED_DIR / "features.parquet")

    return df


if __name__ == "__main__":
    actual_df = pd.read_parquet(_PROCESSED_DIR / "actual_load.parquet")
    temp_df   = pd.read_parquet(_PROCESSED_DIR / "temperature.parquet")

    df = build_features(actual_df, temp_df)

    null_counts = df.isnull().sum()
    print(
        f"features\n"
        f"  shape       : {df.shape}\n"
        f"  date span   : {df.index.min()} to {df.index.max()}\n"
        f"\n  columns & null counts:"
    )
    for col in df.columns:
        print(f"    {col:<30} nulls: {null_counts[col]:>6,}")

    print(f"\nSaved: {_PROCESSED_DIR / 'features.parquet'}")
