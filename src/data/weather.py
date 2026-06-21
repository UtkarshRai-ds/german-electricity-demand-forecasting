from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

_API_URL = "https://archive-api.open-meteo.com/v1/archive"
_LAT = 51.165691
_LON = 10.451526
_START = "2020-01-01"
_TZ = "Europe/Berlin"


def fetch_temperature(start: str = _START, end: str | None = None) -> pd.DataFrame:
    """Fetch hourly temperature_2m for Germany from Open-Meteo and return a UTC DataFrame."""
    if end is None:
        end = date.today().isoformat()

    params = {
        "latitude": _LAT,
        "longitude": _LON,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m",
        "timezone": _TZ,
    }

    response = requests.get(_API_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
    })

    # times arrive as wall-clock Europe/Berlin strings.
    # ambiguous="NaT" marks the duplicate DST fall-back hour as NaT instead of
    # raising; we forward-fill so the earlier (pre-clock-change) value is kept,
    # then drop any residual NaTs before converting to UTC.
    localized = df["timestamp"].dt.tz_localize(
        _TZ, ambiguous="NaT", nonexistent="shift_forward"
    )
    localized = localized.ffill().dropna()
    df = df.loc[localized.index]
    df["timestamp"] = localized.dt.tz_convert("UTC")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df


if __name__ == "__main__":
    df = fetch_temperature()

    n_null = int(df["temperature_2m"].isna().sum())
    print(
        f"temperature_2m\n"
        f"  shape       : {df.shape}\n"
        f"  date span   : {df.index.min()} → {df.index.max()}\n"
        f"  nulls       : {n_null:>10,}\n"
    )

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = _PROCESSED_DIR / "temperature.parquet"
    df.to_parquet(out)
    print(f"Saved → {out}")
