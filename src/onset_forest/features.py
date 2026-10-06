"""Forest input features: discovered from a training frame, then fixed."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

COORDINATE_COLUMNS = ("lat", "lon")
CLIMATOLOGY_PREFIX = "clim_p_onset"


class MissingFeaturesError(KeyError, ValueError):
    """A frame lacks feature columns the forest needs."""

    def __init__(self, missing: Sequence[str], context: str = "Missing feature columns"):
        self.missing = tuple(missing)
        self.message = f"{context}: {', '.join(self.missing)}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


def rain_prefix(model_name: str) -> str:
    return f"{model_name}_rain_mean"


def sorted_day_columns(columns: Iterable[object], prefix: str) -> list[str]:
    """Columns named exactly `{prefix}_day_{k}` for integer k, ordered by k."""
    pattern = re.compile(rf"^{re.escape(prefix)}_day_(\d+)$")
    found = [(int(m.group(1)), str(col)) for col in columns if (m := pattern.match(str(col)))]
    return [col for _, col in sorted(found)]


def climatology_columns(max_day: int) -> tuple[str, ...]:
    return tuple(f"{CLIMATOLOGY_PREFIX}_day_{d}" for d in range(1, max_day + 1))


def _model_rain_columns(columns: Sequence[str], model_name: str) -> list[str]:
    found = sorted_day_columns(columns, rain_prefix(model_name))
    if not found:
        raise MissingFeaturesError(
            [f"{rain_prefix(model_name)}_day_*"], f"No rain columns for model {model_name!r}"
        )
    return found


def _missing(columns: Iterable[str], available: Iterable[object]) -> list[str]:
    present = set(map(str, available))
    return [col for col in columns if col not in present]


@dataclass(frozen=True)
class FeatureSpec:
    columns: tuple[str, ...]

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, model_names: Sequence[str], max_day: int = 28
    ) -> FeatureSpec:
        """lat, lon, each model's rain days ascending, then climatology days 1..max_day."""
        if not model_names:
            raise ValueError("At least one model name is required.")
        fixed = COORDINATE_COLUMNS + climatology_columns(max_day)
        missing = _missing(fixed, frame.columns)
        if missing:
            raise MissingFeaturesError(missing)
        rain = [col for m in model_names for col in _model_rain_columns(list(frame.columns), m)]
        return cls(COORDINATE_COLUMNS + tuple(rain) + climatology_columns(max_day))

    def without(self, columns: Iterable[str]) -> FeatureSpec:
        dropped = set(columns)
        return FeatureSpec(tuple(col for col in self.columns if col not in dropped))

    def observed_in(self, frame: pd.DataFrame) -> FeatureSpec:
        """Drop columns with no observed value in frame (as sklearn's median imputer does)."""
        values = self.frame(frame)
        return self.without(values.columns[values.isna().all()])

    def frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = _missing(self.columns, frame.columns)
        if missing:
            raise MissingFeaturesError(missing)
        return frame.loc[:, list(self.columns)].astype(np.float64)

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        return self.frame(frame).to_numpy(dtype=np.float64)
