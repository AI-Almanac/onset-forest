from __future__ import annotations

import numpy as np
import pytest

from onset_forest.forest import _CRPSTree


def _same_tree(a, b) -> None:
    assert a.is_leaf == b.is_leaf
    if a.is_leaf:
        np.testing.assert_array_equal(a.values, b.values)
        return
    assert (a.feature, a.threshold) == (b.feature, b.threshold)
    _same_tree(a.left, b.left)
    _same_tree(a.right, b.right)


@pytest.mark.parametrize(("max_features", "max_depth"), [("sqrt", None), (None, 4)])
def test_a_tree_grown_from_row_indices_is_the_tree_grown_from_those_rows(
    max_features, max_depth
) -> None:
    rng = np.random.default_rng(0)
    X = np.round(rng.gamma(1.0, 5.0, size=(1500, 12)))  # ties, as daily rainfall has
    y = rng.integers(1, 30, size=1500).astype(np.float64)
    rows = rng.integers(0, 1500, size=1500)  # a bootstrap draw: unordered, with repeats

    def grow():
        return _CRPSTree(
            min_samples_leaf=10, random_state=3, max_depth=max_depth, max_features=max_features
        )

    from_rows = grow().fit_rows(X, y, rows)
    from_copy = grow().fit(X[rows], y[rows])

    _same_tree(from_rows.root_, from_copy.root_)
