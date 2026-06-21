# Feature Leakage Audit

## What was leaking

Three feature groups in `src/features/engineer.py` used information from
hour H to predict load at hour H, violating the 24-hour forecast horizon:

| Feature | Leak |
|---|---|
| `rolling_mean_24h`, `rolling_std_24h`, `rolling_mean_168h` | `load.rolling(w).mean()` at row H includes `load[H]` (the target) |
| `temp_2m` | same-hour temperature — unknown 24 h in advance |
| `temp_rolling_mean_72h` | `temp.rolling(72).mean()` at row H includes `temp[H]` |

`lag_24` and `lag_168` were already clean (shift ≥ horizon), as were all
calendar features (hour, day_of_week, month, is_weekend, is_holiday).

## Fix applied (`horizon: int = 24`)

```python
# Rolling features: shift the series FIRST, then roll
load_shifted = load.shift(horizon)
df["rolling_mean_24h"]  = load_shifted.rolling(24,  min_periods=24).mean()
df["rolling_std_24h"]   = load_shifted.rolling(24,  min_periods=24).std()
df["rolling_mean_168h"] = load_shifted.rolling(168, min_periods=168).mean()

# Temperature: shift everything by horizon
temp_shifted = temp.shift(horizon)
df["temp_2m"]               = temp_shifted
df["temp_lag_24"]           = temp.shift(max(24, horizon))
df["temp_rolling_mean_72h"] = temp_shifted.rolling(72, min_periods=72).mean()
```

## Before vs After — Test MAPE on 2026-01-01 to 2026-06-15 holdout

| Model | Before (leaked) | After (honest) | Delta |
|---|---|---|---|
| LightGBM | 2.7686% | 3.0561% | +0.29 pp |
| XGBoost  | 2.5829% | 3.0801% | +0.50 pp |
| CatBoost | 2.2419% | 2.8321% | +0.59 pp |
| TSO benchmark | 3.80% | 3.80% | — |

All three models still beat the TSO benchmark after the fix.
The gap was artificially inflated by ~0.3–0.6 percentage points.

## Pytest

All 8 tests pass on the regenerated `features.parquet`.
No tests needed modification — column names are unchanged, only the
computation is now correctly shifted.

```
8 passed, 1 warning in 1.47s
```

The warning (`FutureWarning: isin with datetime64`) is pre-existing and
unrelated to this fix.
