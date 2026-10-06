"""Reproduce Experiment 2 of the science team's CRPS random forest notebook.

Experiment 2 ("raw daily inputs with climatology") trains on lat, lon, every
aifs/ngcm daily rain mean and clim_p_onset_day_1..28, with four contiguous 5-year
validation blocks over 2000-2019. The notebook reports a multiclass Brier of ~0.5255.

    uv run python scripts/reproduce_notebook.py \
        --rds combine_template_2025_combined_wide.rds \
        --dissemination-cells dissemination_cells.csv
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from onset_forest import WEEKLY_COLUMNS, ForestParams, fit, predict

MODELS = ("aifs", "ngcm")
CV_YEARS = tuple(range(2000, 2020))
N_BLOCKS = 4
# The notebook keeps only rows whose 36 engineered features are finite. Those features
# use rain days 1..37 of both models (10-day sums starting up to day 28) and the
# onset threshold, so a row survives exactly when all of these are finite.
ENGINEERED_RAIN_DAYS = range(1, 38)
ONSET_THRESHOLD_COLUMN = "ngcm_onset_thresh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rds", type=Path, required=True)
    parser.add_argument("--dissemination-cells", type=Path, required=True, help="CSV with lat,lon")
    parser.add_argument("--n-estimators", type=int, default=ForestParams.n_estimators)
    parser.add_argument("--n-jobs", type=int, default=ForestParams.n_jobs)
    return parser.parse_args()


def load_rds_dataframe(path: Path) -> pd.DataFrame:
    import pyreadr

    result = pyreadr.read_r(str(path))
    if not result:
        raise ValueError(f"No objects found in RDS file: {path}")
    return next(iter(result.values())).copy()


def coordinate_token(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else format(value, "g")


def load_cells(path: Path) -> pd.DataFrame:
    cells = pd.read_csv(path)[["lat", "lon"]].drop_duplicates()
    cells = cells.astype(float)
    cells["id"] = cells["lat"].map(coordinate_token) + "_" + cells["lon"].map(coordinate_token)
    return cells


def to_date(values: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="D", origin="unix", errors="coerce")
    return pd.to_datetime(values, errors="coerce")


def lead_days(frame: pd.DataFrame) -> pd.Series:
    return (to_date(frame["true_onset_date"]) - to_date(frame["time"])).dt.days


def onset_week_class(lead: pd.Series) -> np.ndarray:
    """0..3 for weeks 1..4 (lead 1-7, ..., 22-28); 4 for later or no onset."""
    week = np.ceil(lead.to_numpy(dtype=float) / 7.0) - 1
    return np.where(np.isfinite(week) & (week <= 3), week, 4).astype(int)


def engineered_features_finite(frame: pd.DataFrame) -> pd.Series:
    columns = [f"{m}_rain_mean_day_{d}" for m in MODELS for d in ENGINEERED_RAIN_DAYS]
    values = frame[columns + [ONSET_THRESHOLD_COLUMN]].to_numpy(dtype=float)
    return pd.Series(np.isfinite(values).all(axis=1), index=frame.index)


def notebook_modeling_rows(raw: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """The notebook's modeling sample, with lat/lon taken from the dissemination cells."""
    raw = raw.assign(id=raw["id"].astype(str), year=pd.to_numeric(raw["year"]).astype(int))
    missing_ids = sorted(set(cells["id"]) - set(raw["id"]))
    if missing_ids:
        raise ValueError(f"Dissemination cells not in the RDS data: {', '.join(missing_ids)}")
    rows = raw.loc[raw["id"].isin(cells["id"]) & raw["year"].isin(CV_YEARS)]
    clim = [f"clim_p_onset_day_{d}" for d in range(1, 29)]
    rows = rows.loc[rows[clim].notna().all(axis=1)]
    lead = lead_days(rows)
    rows = rows.loc[~(lead.notna() & (lead <= 0)) & engineered_features_finite(rows)]
    if rows.duplicated(["id", "time"]).any():
        raise ValueError("The RDS data has duplicate (id, time) rows.")
    coords = cells[["id", "lat", "lon"]]
    return rows.drop(columns=["lat", "lon"], errors="ignore").merge(
        coords, on="id", how="left", validate="many_to_one"
    )


def year_blocks() -> list[tuple[int, ...]]:
    return [tuple(int(y) for y in b) for b in np.array_split(np.array(CV_YEARS), N_BLOCKS)]


def block_predictions(rows: pd.DataFrame, params: ForestParams) -> pd.DataFrame:
    """Notebook CV: block fold_id (1-based) trains on the other blocks with seed + fold_id."""
    parts = []
    for fold_id, held_out in enumerate(year_blocks(), start=1):
        is_held_out = rows["year"].isin(held_out)
        fold_params = replace(params, random_state=params.random_state + fold_id)
        fitted = fit(rows.loc[~is_held_out], MODELS, fold_params)
        val = rows.loc[is_held_out]
        parts.append(pd.concat([val, predict(fitted, val)], axis=1).assign(fold_id=fold_id))
    return pd.concat(parts, ignore_index=True)


def brier_losses(predictions: pd.DataFrame) -> np.ndarray:
    probs = predictions[list(WEEKLY_COLUMNS)].to_numpy(dtype=float)
    observed = np.eye(len(WEEKLY_COLUMNS))[onset_week_class(lead_days(predictions))]
    return ((observed - probs) ** 2).sum(axis=1)


def report(predictions: pd.DataFrame) -> None:
    losses = brier_losses(predictions)
    for fold_id, fold_losses in pd.Series(losses).groupby(predictions["fold_id"]):
        print(f"  fold {fold_id}: Brier {fold_losses.mean():.4f} (n={len(fold_losses):,})")
    se = losses.std(ddof=1) / np.sqrt(len(losses))
    print(f"Multiclass Brier: {losses.mean():.4f} (se {se:.4f}, n={len(losses):,})")


def main() -> None:
    args = parse_args()
    cells = load_cells(args.dissemination_cells)
    rows = notebook_modeling_rows(load_rds_dataframe(args.rds), cells)
    print(f"Modeling rows: {len(rows):,}; locations: {rows['id'].nunique()}")
    params = ForestParams(n_estimators=args.n_estimators, n_jobs=args.n_jobs)
    report(block_predictions(rows, params))


if __name__ == "__main__":
    main()
