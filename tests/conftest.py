"""Shared fixtures. Everything here reads committed artefacts; nothing trains a model.

The suite is deliberately built around mutation: the DQ tests corrupt a copy of the
real data and assert that a specific check fires. A test that only asserts "the real
data passes" proves the data is clean, not that the checks work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "src"))

RAW_CSV = BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUTS = BASE / "outputs"


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    import features as F
    return F.load_raw(RAW_CSV)


@pytest.fixture(scope="session")
def engineered(raw_df) -> pd.DataFrame:
    import features as F
    return F.engineer(raw_df)


@pytest.fixture(scope="session")
def scored() -> pd.DataFrame:
    """The committed scored base - the same file the dashboard and SQL layer read."""
    return pd.read_csv(OUTPUTS / "scored_customers.csv")


@pytest.fixture(scope="session")
def calibrated_model():
    """The deployed artefact. Tests fail loudly if it is absent rather than skipping,
    because 'the API scores with a different model than the pipeline saved' is exactly
    the bug this suite exists to catch."""
    import joblib
    return joblib.load(OUTPUTS / "model_calibrated.pkl")


@pytest.fixture
def corrupt(raw_df) -> pd.DataFrame:
    """A copy of the raw frame a test can damage without touching the session fixture."""
    return raw_df.copy()
