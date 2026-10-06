# onset-forest

A CRPS random forest that blends AI weather-model rainfall forecasts and onset
climatology into a **per-day** monsoon onset distribution. Ported from the science
team's notebook (`crps_rf_three_experiments_engineered_raw_with_without_climatology.ipynb`,
Experiment 2: raw daily inputs with climatology). Seeded fits reproduce the notebook's trees.

It is a library: the blending pipeline calls it in-process.

## Lead-day convention

`lead_day = (true_onset_date - time).days`, where `time` is the issue date.

- Day `d` (1..28) is the calendar date issue + d.
- Week `w` covers lead days `7(w-1)+1 .. 7w`: week 1 is issue+1..+7, week 4 is issue+22..+28.
- "Later" is lead day > 28, which includes "no onset" (pseudo lead day 29) under the
  default `missing_onset_policy="after_week4"`. Under `"drop"` those rows are not trained on.
- Training rows whose onset is on or before the issue date are dropped.

The forest regresses the lead day. A row's prediction pools the training lead days in the
leaves it reaches across all trees: `p_day_d` is the share equal to `d`, `p_later` the share
above 28. `cv_week1..4` are 7-day sums of the daily shares and `cv_later == p_later`.

## API

```python
from onset_forest import (
    DAILY_COLUMNS,  # ("p_day_1", ..., "p_day_28", "p_later")
    WEEKLY_COLUMNS,  # ("cv_week1", ..., "cv_week4", "cv_later")
    ForestParams,
    FeatureSpec,
    FittedForest,
    Fold,
    MissingFeaturesError,
    fit,
    predict,
    cross_validate,
    weekly_from_daily,
    to_bytes,
    from_bytes,
    save,
    load,
)

fitted = fit(train, ["aifs", "ngcm"], ForestParams(n_estimators=100))
probs = predict(fitted, frame)  # DAILY_COLUMNS + WEEKLY_COLUMNS, index == frame.index

folds = [(2001, (2000, 2002, 2003)), (2002, (2000, 2001, 2003))]  # (test_year, train_years)
cv_preds = cross_validate(train, predict_on, folds, ["aifs", "ngcm"])

blob = to_bytes(fitted)
fitted = from_bytes(blob)
```

Input columns:

- `time` (issue date), `true_onset_date` (fit only; missing means no onset), `year` (CV only)
- `lat`, `lon`
- `{model}_rain_mean_day_{k}` for each model, every `k` present at fit time
- `clim_p_onset_day_1` .. `clim_p_onset_day_28`

Feature order is lat, lon, each model's rain days ascending, then climatology days.
`fit` fixes the features (dropping any column with no observed value in training) and the
median used to impute missing values. `predict` on a frame that lacks a feature column raises
`MissingFeaturesError` listing the columns.

In `cross_validate`, fold `i` (0-based) is seeded `random_state + i + 1`. It fits on the `train` rows
whose year is in `train_years`, and predicts the `predict_on` rows from `test_year`. Those rows
keep all their original columns.

`from_bytes` and `load` unpickle, so only load artifacts you produced.

The numba kernels cache compiled code next to the package, or in `NUMBA_CACHE_DIR` when the
package directory is read-only. If no location is writable, they compile in memory.

## Development

```bash
uv run pytest -q
uv run ruff check && uv run ruff format --check
```

Reproduce the notebook's Experiment 2 (multiclass Brier ≈ 0.5255 on the dissemination cells):

```bash
uv run python scripts/reproduce_notebook.py \
    --rds combine_template_2025_combined_wide.rds \
    --dissemination-cells dissemination_cells.csv   # columns lat,lon
```
