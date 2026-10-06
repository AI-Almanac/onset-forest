from __future__ import annotations

import numpy as np
from conftest import synthetic_frame

from onset_forest.targets import lead_day_target

LEADS = np.array([-2.0, 0.0, 1.0, 28.0, 35.0, np.nan])


def test_after_week4_keeps_future_onsets_and_maps_no_onset_after_the_horizon():
    target = lead_day_target(synthetic_frame(LEADS, np.full(6, 2001)), "after_week4")
    assert target.to_dict() == {2: 1.0, 3: 28.0, 4: 35.0, 5: 29.0}


def test_drop_policy_discards_rows_without_an_onset():
    target = lead_day_target(synthetic_frame(LEADS, np.full(6, 2001)), "drop")
    assert target.to_dict() == {2: 1.0, 3: 28.0, 4: 35.0}


def test_r_day_counts_are_read_as_dates():
    frame = synthetic_frame(np.array([5.0]), np.array([2001]))
    epoch = np.datetime64("1970-01-01")
    frame["time"] = (frame["time"].to_numpy().astype("datetime64[D]") - epoch).astype(float)
    frame["true_onset_date"] = frame["time"] + 5
    assert lead_day_target(frame).tolist() == [5.0]
