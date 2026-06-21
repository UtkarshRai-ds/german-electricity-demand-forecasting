from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.loader import load_smard
from src.data.quality import check_quality
from src.features.engineer import build_features

_PROJECT_ROOT  = Path(__file__).resolve().parents[1]
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

_TRAIN_END = pd.Timestamp("2026-01-01", tz="UTC")

_EXPECTED_FEATURE_COLS = [
    "actual_load",
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "lag_24", "lag_168",
    "rolling_mean_24h", "rolling_std_24h", "rolling_mean_168h",
    "temp_2m", "temp_lag_24", "temp_rolling_mean_72h",
]


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def actual_df() -> pd.DataFrame:
    return load_smard("Actual_consumption_", "actual_load")


@pytest.fixture(scope="session")
def actual_parquet() -> pd.DataFrame:
    return pd.read_parquet(_PROCESSED_DIR / "actual_load.parquet")


@pytest.fixture(scope="session")
def temp_parquet() -> pd.DataFrame:
    return pd.read_parquet(_PROCESSED_DIR / "temperature.parquet")


@pytest.fixture(scope="session")
def features_df(actual_parquet, temp_parquet) -> pd.DataFrame:
    return build_features(actual_parquet, temp_parquet)


# ── loader tests ───────────────────────────────────────────────────────────────

def test_loader_columns(actual_df):
    assert list(actual_df.columns) == ["actual_load"]
    assert isinstance(actual_df.index, pd.DatetimeIndex)


def test_loader_no_nulls(actual_df):
    assert actual_df["actual_load"].isna().sum() == 0


def test_loader_sorted_index(actual_df):
    assert actual_df.index.is_monotonic_increasing


# ── quality gate tests ─────────────────────────────────────────────────────────

def test_quality_gate_passes_clean_data(actual_parquet):
    report = check_quality(actual_parquet, "actual_load")
    assert report["success"] is True


def test_quality_gate_fails_on_negatives(actual_parquet):
    bad = actual_parquet.copy()
    bad.iloc[0, bad.columns.get_loc("actual_load")] = -999.0
    report = check_quality(bad, "actual_load")
    assert report["success"] is False


# ── feature engineering tests ──────────────────────────────────────────────────

def test_features_no_nulls(features_df):
    assert features_df.isnull().sum().sum() == 0


def test_features_expected_columns(features_df):
    assert list(features_df.columns) == _EXPECTED_FEATURE_COLS


# ── temporal split test ────────────────────────────────────────────────────────

def test_temporal_split_no_leakage(features_df):
    idx = features_df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    train_max = idx[idx < _TRAIN_END].max()
    test_min  = idx[idx >= _TRAIN_END].min()

    assert train_max < test_min
