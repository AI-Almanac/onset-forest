from __future__ import annotations

import pytest

from onset_forest import FeatureSpec, MissingFeaturesError


def test_feature_order_is_coordinates_then_models_by_day_then_climatology(frame):
    shuffled = frame[list(reversed(frame.columns))].assign(aifs_rain_mean_day_10_extra=1.0)
    spec = FeatureSpec.from_frame(shuffled, ["ngcm", "aifs"])
    assert spec.columns[:2] == ("lat", "lon")
    assert spec.columns[2:12] == tuple(f"ngcm_rain_mean_day_{k}" for k in range(1, 11))
    assert spec.columns[12:22] == tuple(f"aifs_rain_mean_day_{k}" for k in range(1, 11))
    assert spec.columns[22:] == tuple(f"clim_p_onset_day_{d}" for d in range(1, 29))


def test_unknown_model_is_rejected(frame):
    with pytest.raises(MissingFeaturesError, match="graphcast"):
        FeatureSpec.from_frame(frame, ["aifs", "graphcast"])


def test_missing_climatology_is_rejected(frame):
    with pytest.raises(MissingFeaturesError, match="clim_p_onset_day_28"):
        FeatureSpec.from_frame(frame.drop(columns=["clim_p_onset_day_28"]), ["aifs"])


def test_missing_features_error_is_a_key_and_value_error():
    assert issubclass(MissingFeaturesError, KeyError)
    assert issubclass(MissingFeaturesError, ValueError)
