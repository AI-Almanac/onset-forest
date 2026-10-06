from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from onset_forest import ForestParams

MODELS = ("aifs", "ngcm")
RAIN_DAYS = 10


def synthetic_frame(
    lead_days: np.ndarray, years: np.ndarray, seed: int = 0, rain_days: int = RAIN_DAYS
) -> pd.DataFrame:
    """One row per lead day; NaN lead day means no onset. Features are noise."""
    rng = np.random.default_rng(seed)
    n = len(lead_days)
    issue = pd.Timestamp("2010-05-01") + pd.to_timedelta(rng.integers(0, 30, size=n), unit="D")
    onset = issue + pd.to_timedelta(lead_days, unit="D")
    columns = {
        "id": [f"cell_{i % 5}" for i in range(n)],
        "year": years.astype(int),
        "time": issue,
        "true_onset_date": onset,
        "lat": rng.uniform(5, 15, size=n),
        "lon": rng.uniform(70, 80, size=n),
    }
    for model in MODELS:
        for k in range(1, rain_days + 1):
            columns[f"{model}_rain_mean_day_{k}"] = rng.gamma(2.0, 3.0, size=n)
    for d in range(1, 29):
        columns[f"clim_p_onset_day_{d}"] = rng.uniform(0, 0.05, size=n)
    return pd.DataFrame(columns)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 300
    lead = rng.integers(-3, 40, size=n).astype(float)
    lead[rng.random(n) < 0.1] = np.nan
    years = rng.choice([2001, 2002, 2003], size=n)
    return synthetic_frame(lead, years)


@pytest.fixture
def params() -> ForestParams:
    return ForestParams(n_estimators=5, min_samples_leaf=10, n_jobs=1)
