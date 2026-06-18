"""Time-series-safe cross-validation splitters for DoubleML."""

from collections.abc import Iterator
import numpy as np


class PurgedEmbargoedKFold:
    """
    K-Fold splitter for time series data.

    Splits indices without shuffling. Around each test
    fold, `embargo` rows are excluded from training on both ends; the expectation is
    the short-range autocorrelation will be eliminated (may need ljung-box )
    """

    def __init__(self, n_splits: int = 3, embargo: int = 2, purge_window: int = 0):
        self.n_splits = n_splits
        self.embargo = embargo
        self.purge_window = purge_window

    def split(self, X, y=None, groups=None) -> Iterator[(np.ndarray, np.ndarray)]:
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end = (k + 1) * fold_size if k < self.n_splits - 1 else n
            test_idx = indices[test_start:test_end]

            lo = max(0, test_start - self.embargo - self.purge_window)
            hi = min(n, test_end + self.embargo + self.purge_window)

            train_mask = np.ones(n, dtype=bool)
            train_mask[lo:hi] = False  # purge + embargo zone
            train_idx = indices[train_mask]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
