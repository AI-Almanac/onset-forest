from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import MODELS, synthetic_frame

from onset_forest import DAILY_COLUMNS, WEEKLY_COLUMNS, cross_validate


def year_marked_frame():
    """Onset is day 5 in 2001 only; every other year onsets on day 20."""
    years = np.repeat([2000, 2001, 2002, 2003], 50)
    leads = np.where(years == 2001, 5.0, 20.0)
    return synthetic_frame(leads, years, seed=3)


def test_predicts_only_test_year_rows_and_keeps_their_columns(params):
    data = year_marked_frame()
    predict_on = data.assign(extra="kept")
    out = cross_validate(
        data, predict_on, [(2001, (2000, 2002)), (2003, (2000, 2002))], MODELS, params
    )
    assert sorted(out["year"].unique()) == [2001, 2003]
    assert len(out) == int(predict_on["year"].isin([2001, 2003]).sum())
    assert set(predict_on.columns) | set(DAILY_COLUMNS) | set(WEEKLY_COLUMNS) == set(out.columns)
    assert (out["extra"] == "kept").all()
    assert out.index.equals(pd.RangeIndex(len(out)))


def test_test_year_is_never_seen_in_training(params):
    data = year_marked_frame()
    folds = [(2001, (2000, 2002, 2003)), (2002, (2000, 2001, 2003))]
    out = cross_validate(data, data, folds, MODELS, params)
    assert (out.loc[out["year"] == 2001, "p_day_5"] == 0.0).all()
    assert (out.loc[out["year"] == 2002, "p_day_5"] > 0.0).any()


def test_fold_that_trains_on_its_test_year_is_rejected(params):
    data = year_marked_frame()
    with pytest.raises(ValueError, match="2001"):
        cross_validate(data, data, [(2001, (2000, 2001))], MODELS, params)
