"""Training target: onset lead day relative to the issue date."""

from __future__ import annotations

import pandas as pd

ONSET_DATE_COLUMN = "true_onset_date"
ISSUE_DATE_COLUMN = "time"


def coerce_date(values: pd.Series) -> pd.Series:
    """Datetimes from datetime, ISO-string, or R day-count (days since 1970-01-01) columns."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values)
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="D", origin="unix", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def lead_days(frame: pd.DataFrame) -> pd.Series:
    """(true_onset_date - time) in whole days; NaN where there was no onset."""
    onset = coerce_date(frame[ONSET_DATE_COLUMN])
    issue = coerce_date(frame[ISSUE_DATE_COLUMN])
    return (onset - issue).dt.days.astype(float)


def lead_day_target(
    frame: pd.DataFrame, missing_onset_policy: str = "after_week4", max_day: int = 28
) -> pd.Series:
    """Lead-day target for the rows the forest trains on, indexed like those rows.

    Rows whose onset is on or before the issue date are dropped. Rows without an onset
    become the pseudo lead day max_day + 1 under "after_week4" and are dropped under
    "drop".
    """
    lead = lead_days(frame)
    if missing_onset_policy == "after_week4":
        lead = lead.fillna(float(max_day + 1))
    elif missing_onset_policy != "drop":
        raise ValueError("missing_onset_policy must be 'after_week4' or 'drop'.")
    return lead[lead.notna() & (lead > 0)].rename("lead_day")
