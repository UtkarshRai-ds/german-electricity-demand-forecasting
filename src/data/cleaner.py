from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.quality import _print_report, check_quality

_PROJECT_ROOT  = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


def clean_series(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return a gap-free, interpolated copy of *df* and write it to data/processed/."""
    # ── 1. Reindex to a complete hourly grid ──────────────────────────────────
    # This surfaces DST spring-forward gaps (typically 7 per year for CET→CEST)
    # as NaN rows alongside any in-place NaNs already in the data.
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        name=df.index.name,
    )
    df = df.reindex(full_range)

    # ── 2. Time-aware interpolation ───────────────────────────────────────────
    # method="time" weights each gap by its actual duration in seconds, so a
    # 2-hour DST hole is bridged proportionally rather than using equal steps.
    df[name] = df[name].interpolate(method="time")

    # ── 3. Hard assertion: no residual NaNs ───────────────────────────────────
    # Leading / trailing NaNs cannot be interpolated and indicate a structural
    # problem with the source data — surface it immediately rather than silently.
    n_null = int(df[name].isna().sum())
    assert n_null == 0, (
        f"{n_null:,} NaN(s) remain in '{name}' after interpolation. "
        "Check for leading/trailing gaps at the series boundary."
    )

    # ── 4. Post-clean quality gate ────────────────────────────────────────────
    report = check_quality(df, name)
    if not report["success"]:
        raise ValueError(
            f"Quality gate failed on cleaned '{name}':\n"
            + "\n".join(f"  {msg}" for msg in report["failures"])
        )

    # ── 5. Persist ────────────────────────────────────────────────────────────
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PROCESSED_DIR / f"{name}.parquet")

    return df


if __name__ == "__main__":
    from src.data.loader import load_smard

    pairs = [
        ("Actual_consumption_",     "actual_load"),
        ("Forecasted_consumption_", "forecast_load"),
    ]

    for prefix, name in pairs:
        raw = load_smard(prefix, name)

        rows_before = len(raw)
        null_before = int(raw[name].isna().sum())

        cleaned = clean_series(raw, name)

        rows_after = len(cleaned)
        null_after = int(cleaned[name].isna().sum())

        report = check_quality(cleaned, name)

        print(f"\n{'═' * 60}")
        print(f"  {name}")
        print(f"{'═' * 60}")
        print(f"  rows  : {rows_before:>10,}  →  {rows_after:>10,}")
        print(f"  nulls : {null_before:>10,}  →  {null_after:>10,}")
        _print_report(name, report)
