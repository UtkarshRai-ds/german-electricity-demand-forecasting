from __future__ import annotations

from typing import Any

import pandas as pd

# Thresholds
_MISSING_WARN = 0.001   # 0.1 % of expected hourly slots
_MISSING_FAIL = 0.01    # 1.0 %
_NULL_WARN    = 0.005   # 0.5 % of rows
_NULL_FAIL    = 0.05    # 5.0 %
_LOAD_MIN     = 10_000  # MWh — German grid lower plausibility bound
_LOAD_MAX     = 100_000 # MWh — German grid upper plausibility bound


def check_quality(df: pd.DataFrame, name: str) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    series = df[name]

    # ── 1. Index sanity ───────────────────────────────────────────────────────
    if not isinstance(df.index, pd.DatetimeIndex):
        failures.append(f"index is {type(df.index).__name__}, expected DatetimeIndex")
    else:
        if not df.index.is_monotonic_increasing:
            failures.append("index is not sorted in ascending order")
        n_dup = int(df.index.duplicated().sum())
        if n_dup:
            failures.append(f"index has {n_dup:,} duplicate timestamps")

    # ── 2. Hourly continuity ──────────────────────────────────────────────────
    n_missing = 0
    if isinstance(df.index, pd.DatetimeIndex) and len(df):
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        n_expected = len(full_range)
        n_missing  = int(len(full_range.difference(df.index)))
        rate       = n_missing / n_expected if n_expected else 0.0

        if rate > _MISSING_FAIL:
            failures.append(
                f"hourly continuity: {n_missing:,} missing slots "
                f"({rate:.2%} exceeds {_MISSING_FAIL:.0%} failure threshold)"
            )
        elif rate > _MISSING_WARN:
            warnings.append(
                f"hourly continuity: {n_missing:,} missing slots "
                f"({rate:.2%} exceeds {_MISSING_WARN:.1%} warning threshold)"
            )

    # ── 3. Numeric dtype + no negatives ───────────────────────────────────────
    if not pd.api.types.is_numeric_dtype(series):
        failures.append(f"'{name}' is not numeric (dtype={series.dtype})")
    else:
        n_neg = int((series < 0).sum())
        if n_neg:
            failures.append(f"'{name}' contains {n_neg:,} negative value(s)")

    # ── 4. Plausible German load range ────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(series):
        n_out = int(((series < _LOAD_MIN) | (series > _LOAD_MAX)).sum())
        if n_out:
            warnings.append(
                f"{n_out:,} value(s) outside plausible range "
                f"[{_LOAD_MIN:,}–{_LOAD_MAX:,}] MWh"
            )

    # ── 5. In-place NaNs ──────────────────────────────────────────────────────
    n_null    = int(series.isna().sum())
    null_rate = n_null / len(series) if len(series) else 0.0

    if null_rate > _NULL_FAIL:
        failures.append(
            f"{n_null:,} null value(s) ({null_rate:.2%} exceeds "
            f"{_NULL_FAIL:.0%} failure threshold)"
        )
    elif null_rate > _NULL_WARN:
        warnings.append(
            f"{n_null:,} null value(s) ({null_rate:.2%} exceeds "
            f"{_NULL_WARN:.1%} warning threshold)"
        )

    # ── Statistics ────────────────────────────────────────────────────────────
    valid = series.dropna()
    statistics: dict[str, Any] = {
        "total_rows":     len(df),
        "span_start":     df.index.min().isoformat() if isinstance(df.index, pd.DatetimeIndex) and len(df) else None,
        "span_end":       df.index.max().isoformat() if isinstance(df.index, pd.DatetimeIndex) and len(df) else None,
        "n_missing_hours": n_missing,
        "n_null_values":  n_null,
        "min":            round(float(valid.min()),  2) if len(valid) else None,
        "max":            round(float(valid.max()),  2) if len(valid) else None,
        "mean":           round(float(valid.mean()), 2) if len(valid) else None,
    }

    return {
        "success":    len(failures) == 0,
        "failures":   failures,
        "warnings":   warnings,
        "statistics": statistics,
    }


# ── CLI helper ─────────────────────────────────────────────────────────────────

def _print_report(name: str, report: dict[str, Any]) -> None:
    status = "PASS" if report["success"] else "FAIL"
    print(f"\n{'─' * 60}")
    print(f"  {name}  [{status}]")
    print(f"{'─' * 60}")

    for msg in report["failures"]:
        print(f"  [FAIL]  {msg}")
    for msg in report["warnings"]:
        print(f"  [WARN]  {msg}")
    if not report["failures"] and not report["warnings"]:
        print("  No issues found.")

    s = report["statistics"]
    print(
        f"\n  rows          : {s['total_rows']:,}\n"
        f"  span          : {s['span_start']}  →  {s['span_end']}\n"
        f"  missing hours : {s['n_missing_hours']:,}\n"
        f"  null values   : {s['n_null_values']:,}\n"
        f"  min / mean / max : {s['min']:,.1f} / {s['mean']:,.1f} / {s['max']:,.1f} MWh"
    )


if __name__ == "__main__":
    from src.data.loader import load_smard

    pairs = [
        ("Actual_consumption_",    "actual_load"),
        ("Forecasted_consumption_", "forecast_load"),
    ]

    all_passed = True
    for prefix, name in pairs:
        df     = load_smard(prefix, name)
        report = check_quality(df, name)
        _print_report(name, report)
        all_passed = all_passed and report["success"]

    print(f"\n{'═' * 60}")
    print(f"  Overall gate: {'PASSED' if all_passed else 'FAILED'}")
    print(f"{'═' * 60}\n")
