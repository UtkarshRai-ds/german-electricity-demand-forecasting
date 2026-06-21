from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

_LOAD_COL = "grid load [MWh] Calculated resolutions"
_DATE_COL = "Start date"
_DATE_FMT = "%b %d, %Y %I:%M %p"


def load_smard(prefix: str, name: str) -> pd.DataFrame:
    """Load and concatenate SMARD CSVs matching *prefix*, returning a clean hourly Series-like DataFrame."""
    files = sorted(_RAW_DIR.glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{prefix}*.csv' in {_RAW_DIR}"
        )

    frames = [
        pd.read_csv(
            f,
            sep=";",
            thousands=",",
            decimal=".",
            encoding="utf-8-sig",
            na_values=["-"],
        )
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)

    df["timestamp"] = pd.to_datetime(df[_DATE_COL], format=_DATE_FMT)
    df = df[["timestamp", _LOAD_COL]].rename(columns={_LOAD_COL: name})

    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


if __name__ == "__main__":
    datasets = [
        load_smard("Actual_consumption_", "actual_load"),
        load_smard("Forecasted_consumption_", "forecast_load"),
    ]

    for df in datasets:
        col = df.columns[0]
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
        missing = len(full_range.difference(df.index))
        print(
            f"{col}\n"
            f"  rows        : {len(df):>10,}\n"
            f"  dtype       : {df[col].dtype}\n"
            f"  date span   : {df.index.min()} → {df.index.max()}\n"
            f"  missing hrs : {missing:>10,}\n"
        )
