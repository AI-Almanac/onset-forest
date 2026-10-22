"""Fit the onset forest and turn its predictions into daily and weekly probabilities."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike

import joblib
import numpy as np
import pandas as pd

from ._version import __version__
from .columns import daily_columns, weekly_columns
from .features import FeatureSpec
from .forest import CRPSRandomForestRegressor
from .params import ForestParams
from .targets import lead_day_target


@dataclass(frozen=True)
class FittedForest:
    feature_spec: FeatureSpec
    medians: pd.Series
    forest: CRPSRandomForestRegressor
    params: ForestParams
    version: str


def _regressor(params: ForestParams) -> CRPSRandomForestRegressor:
    return CRPSRandomForestRegressor(
        n_estimators=params.n_estimators,
        min_samples_leaf=params.min_samples_leaf,
        max_features=params.max_features,
        max_samples=params.max_samples,
        max_depth=params.max_depth,
        random_state=params.random_state,
        n_jobs=params.n_jobs,
    )


def _imputed_matrix(fitted: FittedForest, frame: pd.DataFrame) -> np.ndarray:
    features = fitted.feature_spec.frame(frame)
    return features.fillna(fitted.medians).to_numpy(dtype=np.float64)


def fit(
    train: pd.DataFrame, model_names: Sequence[str], params: ForestParams = ForestParams()
) -> FittedForest:
    """Fit on rows with a future (or missing, per policy) onset; impute with their medians."""
    train = train.reset_index(drop=True)
    target = lead_day_target(train, params.missing_onset_policy, params.max_day)
    if target.empty:
        raise ValueError("No training rows have an onset after their issue date.")
    rows = train.loc[target.index]
    spec = FeatureSpec.from_frame(rows, model_names, params.max_day).observed_in(rows)
    features = spec.frame(rows)
    medians = features.median()
    forest = _regressor(params).fit(
        features.fillna(medians).to_numpy(dtype=np.float64), target.to_numpy(dtype=np.float64)
    )
    return FittedForest(spec, medians, forest, params, __version__)


def weekly_from_daily(daily: pd.DataFrame, days_per_week: int = 7) -> pd.DataFrame:
    """Sum consecutive days into weeks (week w = lead days 7(w-1)+1 .. 7w); later stays later."""
    values = daily.to_numpy(dtype=np.float64)
    n_days = values.shape[1] - 1
    if n_days % days_per_week != 0:
        raise ValueError(f"{n_days} daily columns do not divide into {days_per_week}-day weeks.")
    n_weeks = n_days // days_per_week
    weeks = values[:, :n_days].reshape(len(values), n_weeks, days_per_week).sum(axis=2)
    return pd.DataFrame(
        np.column_stack([weeks, values[:, n_days]]),
        index=daily.index,
        columns=list(weekly_columns(n_weeks)),
    )


def predict(fitted: FittedForest, frame: pd.DataFrame) -> pd.DataFrame:
    """Daily onset probabilities (p_day_*, p_later) and their weekly sums (cv_week*, cv_later)."""
    params = fitted.params
    distribution = fitted.forest.predict_lead_day_distribution(
        _imputed_matrix(fitted, frame), max_day=params.max_day
    )
    daily = pd.DataFrame(
        distribution, index=frame.index, columns=list(daily_columns(params.max_day))
    )
    return pd.concat([daily, weekly_from_daily(daily, params.days_per_week)], axis=1)


def to_bytes(fitted: FittedForest) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(fitted, buffer)
    return buffer.getvalue()


def _as_fitted(obj: object) -> FittedForest:
    if not isinstance(obj, FittedForest):
        raise TypeError(f"Expected a FittedForest, got {type(obj).__name__}.")
    return obj


def from_bytes(data: bytes) -> FittedForest:
    """Load a FittedForest. Unpickles: only pass bytes from a trusted source."""
    return _as_fitted(joblib.load(io.BytesIO(data)))


def save(fitted: FittedForest, path: str | PathLike[str]) -> None:
    joblib.dump(fitted, path)


def load(path: str | PathLike[str]) -> FittedForest:
    """Load a FittedForest. Unpickles: only load files from a trusted source."""
    return _as_fitted(joblib.load(path))
