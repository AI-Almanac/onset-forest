from __future__ import annotations

from dataclasses import dataclass

MISSING_ONSET_POLICIES = ("after_week4", "drop")
MAX_FEATURES_CHOICES = ("sqrt", None)


@dataclass(frozen=True)
class ForestParams:
    n_estimators: int = 100
    min_samples_leaf: int = 20
    max_features: str | None = "sqrt"
    max_samples: float = 1.0
    max_depth: int | None = None
    random_state: int = 123
    n_jobs: int = -1
    max_day: int = 28
    days_per_week: int = 7
    missing_onset_policy: str = "after_week4"

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be at least 1, got {self.n_estimators}.")
        if self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be at least 1, got {self.min_samples_leaf}.")
        if self.max_features not in MAX_FEATURES_CHOICES:
            raise ValueError(f'max_features must be "sqrt" or None, got {self.max_features!r}.')
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be at least 1 or None, got {self.max_depth}.")
        if not 0.0 < self.max_samples <= 1.0:
            raise ValueError(f"max_samples must be in (0, 1], got {self.max_samples}.")
        if self.days_per_week < 1 or self.max_day % self.days_per_week != 0:
            raise ValueError(
                f"max_day ({self.max_day}) must be a positive multiple of "
                f"days_per_week ({self.days_per_week})."
            )
        if self.missing_onset_policy not in MISSING_ONSET_POLICIES:
            raise ValueError(
                f"missing_onset_policy must be one of {MISSING_ONSET_POLICIES}, "
                f"got {self.missing_onset_policy!r}."
            )

    @property
    def n_weeks(self) -> int:
        return self.max_day // self.days_per_week
