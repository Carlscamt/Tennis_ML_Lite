"""
Time-series cross-validation for betting model evaluation.

Provides expanding/rolling window CV with configurable gap days
to prevent information leakage around tournament boundaries.
"""
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import polars as pl


logger = logging.getLogger(__name__)


@dataclass
class FoldInfo:
    """Metadata about a single CV fold."""
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int


@dataclass
class TimeSeriesBettingCV:
    """
    Time-series cross-validation for betting models.
    
    Features:
    - Expanding (default) or rolling window training
    - Configurable gap between train/test to prevent leakage
    - Minimum training size enforcement
    - Tournament-aware boundaries (optional)
    
    Example:
        cv = TimeSeriesBettingCV(n_splits=5, gap_days=7)
        for train_idx, test_idx in cv.split(df, date_col="match_date"):
            train_df = df[train_idx]
            test_df = df[test_idx]
            # Train and evaluate...
    """
    n_splits: int = 5
    gap_days: int = 7           # Days between train end and test start
    min_train_size: int = 5000  # Minimum samples in training set
    rolling_window_days: int | None = None  # If set, use rolling window of this size

    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.gap_days < 0:
            raise ValueError("gap_days must be non-negative")

    def split(
        self,
        df: pl.DataFrame,
        date_col: str = "match_date"
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.
        
        Args:
            df: DataFrame with a date column (must be sorted by date)
            date_col: Name of the date column
            
        Yields:
            Tuple of (train_indices, test_indices) for each fold
        """
        # Ensure df is sorted by date
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")

        # Materialize if lazy
        if hasattr(df, 'collect'):
            df = df.collect()

        df = df.sort(date_col)
        n_samples = len(df)

        if n_samples < self.min_train_size:
            raise ValueError(
                f"Dataset too small ({n_samples} samples). "
                f"Need at least {self.min_train_size} for minimum train size."
            )

        # Get date values as numpy array for efficient indexing
        dates = df.select(pl.col(date_col)).to_series().to_numpy()

        # Calculate test fold size (roughly equal sizes)
        # Reserve min_train_size for initial training
        available_for_testing = n_samples - self.min_train_size
        test_fold_size = max(1, available_for_testing // self.n_splits)

        indices = np.arange(n_samples)

        for fold_idx in range(self.n_splits):
            # Calculate test boundaries
            test_start_idx = self.min_train_size + (fold_idx * test_fold_size)
            test_end_idx = min(test_start_idx + test_fold_size, n_samples)

            if test_start_idx >= n_samples:
                logger.warning(f"Fold {fold_idx}: Not enough data, skipping")
                continue

            # Apply gap: move test start forward by gap_days
            if self.gap_days > 0 and test_start_idx > 0:
                test_start_date = dates[test_start_idx]
                # Find first index where date > train_end + gap
                train_end_idx = test_start_idx - 1
                train_end_date = dates[train_end_idx]

                # Calculate gap in terms of actual dates
                # For simplicity, we'll use index-based gap calculation
                # Could be enhanced to use actual calendar days
                gap_samples = self._estimate_gap_samples(dates, train_end_idx, self.gap_days)
                test_start_idx = min(test_start_idx + gap_samples, n_samples - 1)

            if test_start_idx >= test_end_idx:
                logger.warning(f"Fold {fold_idx}: Test set empty after gap, skipping")
                continue

            # Calculate training boundaries
            if self.rolling_window_days is not None:
                # Rolling window: fixed-size training window
                window_samples = self._estimate_gap_samples(
                    dates, test_start_idx, self.rolling_window_days
                )
                train_start_idx = max(0, test_start_idx - window_samples)
            else:
                # Expanding window: all data up to gap
                train_start_idx = 0

            train_end_idx = test_start_idx - 1

            # Ensure minimum training size
            if (train_end_idx - train_start_idx + 1) < self.min_train_size:
                logger.warning(
                    f"Fold {fold_idx}: Training set too small "
                    f"({train_end_idx - train_start_idx + 1} < {self.min_train_size}), skipping"
                )
                continue

            train_indices = indices[train_start_idx:train_end_idx + 1]
            test_indices = indices[test_start_idx:test_end_idx]

            logger.debug(
                f"Fold {fold_idx}: train[{train_start_idx}:{train_end_idx}] "
                f"({len(train_indices)} samples), "
                f"test[{test_start_idx}:{test_end_idx}] ({len(test_indices)} samples)"
            )

            yield train_indices, test_indices

    def _estimate_gap_samples(
        self,
        dates: np.ndarray,
        reference_idx: int,
        days: int
    ) -> int:
        """
        Estimate number of samples corresponding to a given number of days.
        
        Uses average matches per day from recent data.
        """
        if reference_idx == 0 or days == 0:
            return 0

        # Look at last 1000 samples to estimate matches per day
        lookback = min(1000, reference_idx)
        start_idx = reference_idx - lookback

        start_date = dates[start_idx]
        end_date = dates[reference_idx]

        # Handle different date types
        try:
            if hasattr(start_date, 'days'):
                # timedelta-like
                date_diff = (end_date - start_date).days
            else:
                # Assume numeric (days since epoch or similar)
                date_diff = float(end_date - start_date)
        except Exception:
            # Fallback: assume 10 matches per day average
            date_diff = lookback / 10

        if date_diff <= 0:
            return days * 10  # Fallback

        matches_per_day = lookback / max(date_diff, 1)
        return int(days * matches_per_day)

    def get_fold_info(
        self,
        df: pl.DataFrame,
        date_col: str = "match_date"
    ) -> list[FoldInfo]:
        """
        Get metadata about each fold without generating indices.
        
        Useful for understanding the CV structure before training.
        """
        if hasattr(df, 'collect'):
            df = df.collect()

        df = df.sort(date_col)
        dates = df.select(pl.col(date_col)).to_series()

        fold_infos = []
        for fold_idx, (train_idx, test_idx) in enumerate(self.split(df, date_col)):
            # Convert numpy int64 to Python int for Polars indexing
            fold_infos.append(FoldInfo(
                fold_idx=fold_idx,
                train_start=str(dates[int(train_idx[0])]),
                train_end=str(dates[int(train_idx[-1])]),
                test_start=str(dates[int(test_idx[0])]),
                test_end=str(dates[int(test_idx[-1])]),
                train_size=len(train_idx),
                test_size=len(test_idx),
            ))

        return fold_infos


def get_cv_splitter(
    n_splits: int = 5,
    gap_days: int = 7,
    min_train_size: int = 5000,
    rolling_window_days: int | None = None
) -> TimeSeriesBettingCV:
    """
    Factory function to create CV splitter with validation.
    
    Args:
        n_splits: Number of folds
        gap_days: Gap between train and test to prevent leakage
        min_train_size: Minimum training samples required
        rolling_window_days: If set, use rolling window instead of expanding
        
    Returns:
        Configured TimeSeriesBettingCV instance
    """
    return TimeSeriesBettingCV(
        n_splits=n_splits,
        gap_days=gap_days,
        min_train_size=min_train_size,
        rolling_window_days=rolling_window_days
    )
