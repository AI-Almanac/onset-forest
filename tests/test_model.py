from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import MODELS, synthetic_frame

from onset_forest import (
    DAILY_COLUMNS,
    WEEKLY_COLUMNS,
    MissingFeaturesError,
    fit,
    from_bytes,
    load,
    predict,
    save,
    to_bytes,
    weekly_from_daily,
)
from onset_forest.forest import lead_day_histogram


def test_prediction_has_daily_and_weekly_columns_on_the_input_index(frame, params):
    query = frame.set_index(frame.index + 1000)
    preds = predict(fit(frame, MODELS, params), query)
    assert list(preds.columns) == list(DAILY_COLUMNS + WEEKLY_COLUMNS)
    assert preds.index.equals(query.index)


def test_daily_distribution_rows_sum_to_one(frame, params):
    preds = predict(fit(frame, MODELS, params), frame)
    np.testing.assert_allclose(preds[list(DAILY_COLUMNS)].sum(axis=1), 1.0)
    np.testing.assert_allclose(preds[list(WEEKLY_COLUMNS)].sum(axis=1), 1.0)


def test_distribution_matches_pooled_leaf_members(frame, params):
    fitted = fit(frame, MODELS, params)
    X = fitted.feature_spec.frame(frame).fillna(fitted.medians).to_numpy()
    expected = np.array([lead_day_histogram(m, 28) for m in fitted.forest.predict_members(X)])
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(predict(fitted, frame)[list(DAILY_COLUMNS)], expected)


@pytest.mark.parametrize(
    ("lead_day", "column"),
    [
        (1, "cv_week1"),
        (7, "cv_week1"),
        (8, "cv_week2"),
        (22, "cv_week4"),
        (28, "cv_week4"),
        (29, "cv_later"),
        (35, "cv_later"),
    ],
)
def test_constant_lead_day_lands_in_its_week(lead_day, column, params):
    train = synthetic_frame(np.full(60, float(lead_day)), np.full(60, 2001))
    preds = predict(fit(train, MODELS, params), train.head(3))
    assert (preds[column] == 1.0).all()
    day_column = f"p_day_{lead_day}" if lead_day <= 28 else "p_later"
    assert (preds[day_column] == 1.0).all()


def test_weekly_sums_seven_day_blocks_and_keeps_later():
    daily = pd.DataFrame(np.eye(29), columns=list(DAILY_COLUMNS))
    weekly = weekly_from_daily(daily)
    week_of_day = weekly.to_numpy().argmax(axis=1)
    assert list(week_of_day[:7]) == [0] * 7
    assert list(week_of_day[7:14]) == [1] * 7
    assert week_of_day[27] == 3
    assert week_of_day[28] == 4


def test_seeded_fit_is_deterministic_across_thread_counts(frame, params):
    first = predict(fit(frame, MODELS, params), frame)
    again = predict(fit(frame, MODELS, replace(params, n_jobs=2)), frame)
    pd.testing.assert_frame_equal(first, again)


def test_different_seed_changes_the_fit(frame, params):
    first = predict(fit(frame, MODELS, params), frame)
    other = predict(fit(frame, MODELS, replace(params, random_state=999)), frame)
    assert not np.allclose(first.to_numpy(), other.to_numpy())


def test_bytes_round_trip_predicts_identically(frame, params):
    fitted = fit(frame, MODELS, params)
    pd.testing.assert_frame_equal(
        predict(from_bytes(to_bytes(fitted)), frame), predict(fitted, frame)
    )


def test_file_round_trip_predicts_identically(frame, params, tmp_path):
    fitted = fit(frame, MODELS, params)
    save(fitted, tmp_path / "forest.joblib")
    pd.testing.assert_frame_equal(
        predict(load(tmp_path / "forest.joblib"), frame), predict(fitted, frame)
    )


def test_from_bytes_rejects_other_objects():
    import io

    import joblib  # noqa: E401

    buffer = io.BytesIO()
    joblib.dump({"not": "a forest"}, buffer)
    with pytest.raises(TypeError):
        from_bytes(buffer.getvalue())


def test_predict_on_frame_missing_features_lists_them(frame, params):
    fitted = fit(frame, MODELS, params)
    short = frame.drop(columns=["ngcm_rain_mean_day_10", "lat"])
    with pytest.raises(MissingFeaturesError) as err:
        predict(fitted, short)
    assert set(err.value.missing) == {"ngcm_rain_mean_day_10", "lat"}
    assert "ngcm_rain_mean_day_10" in str(err.value)


def test_missing_feature_values_are_imputed(frame, params):
    fitted = fit(frame, MODELS, params)
    gappy = frame.copy()
    gappy.loc[:10, "aifs_rain_mean_day_3"] = np.nan
    preds = predict(fitted, gappy)
    assert preds.notna().all().all()


def test_features_entirely_missing_in_training_are_not_required(frame, params):
    train = frame.assign(aifs_rain_mean_day_10=np.nan)
    fitted = fit(train, MODELS, params)
    assert "aifs_rain_mean_day_10" not in fitted.feature_spec.columns
    predict(fitted, frame.drop(columns=["aifs_rain_mean_day_10"]))
