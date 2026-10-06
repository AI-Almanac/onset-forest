"""Cross-validated forest predictions over caller-defined year folds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pandas as pd

from .model import fit, predict
from .params import ForestParams

Fold = tuple[int, tuple[int, ...]]

YEAR_COLUMN = "year"


def _check_fold(fold: Fold) -> None:
    test_year, train_years = fold
    if test_year in set(train_years):
        raise ValueError(f"Fold for test year {test_year} also trains on {test_year}.")


def _fold_predictions(
    train: pd.DataFrame,
    predict_on: pd.DataFrame,
    fold: Fold,
    model_names: Sequence[str],
    params: ForestParams,
) -> pd.DataFrame:
    test_year, train_years = fold
    rows = predict_on.loc[predict_on[YEAR_COLUMN] == test_year]
    if rows.empty:
        return rows
    fitted = fit(train.loc[train[YEAR_COLUMN].isin(train_years)], model_names, params)
    return pd.concat([rows, predict(fitted, rows)], axis=1)


def cross_validate(
    train: pd.DataFrame,
    predict_on: pd.DataFrame,
    folds: Sequence[Fold],
    model_names: Sequence[str],
    params: ForestParams = ForestParams(),
) -> pd.DataFrame:
    """Predict each fold's test-year rows of predict_on with a forest fit on its train years.

    Fold i (0-based) uses random_state + i + 1. Returned rows keep every predict_on column.
    """
    for fold in folds:
        _check_fold(fold)
    parts = [
        _fold_predictions(
            train,
            predict_on,
            fold,
            model_names,
            replace(params, random_state=params.random_state + i + 1),
        )
        for i, fold in enumerate(folds)
    ]
    predicted = [part for part in parts if not part.empty]
    if not predicted:
        raise ValueError("No predict_on rows fall in any fold's test year.")
    return pd.concat(predicted, ignore_index=True)
