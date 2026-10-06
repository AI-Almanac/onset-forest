"""Names of the prediction columns."""

from __future__ import annotations


def daily_columns(max_day: int = 28) -> tuple[str, ...]:
    return tuple(f"p_day_{d}" for d in range(1, max_day + 1)) + ("p_later",)


def weekly_columns(n_weeks: int = 4) -> tuple[str, ...]:
    return tuple(f"cv_week{w}" for w in range(1, n_weeks + 1)) + ("cv_later",)


DAILY_COLUMNS = daily_columns()
WEEKLY_COLUMNS = weekly_columns()
