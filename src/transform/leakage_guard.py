"""
Data leakage prevention utilities.
Critical for ensuring valid ML evaluation.
"""
import polars as pl
from datetime import date, datetime
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LeakageError(Exception):
    """Raised when data leakage is detected."""
    pass


def validate_temporal_order(df: pl.LazyFrame, timestamp_col: str = "start_timestamp") -> bool:
    """
    Verify that data is sorted by timestamp.
    
    Args:
        df: LazyFrame to check
        timestamp_col: Column containing timestamps
        
    Returns:
        True if sorted
        
    Raises:
        LeakageError if not sorted
    """
    # Check if sorted by comparing consecutive values
    check = df.select([
        (pl.col(timestamp_col).diff() >= 0).all().alias("is_sorted")
    ]).collect()
    
    # First diff is null, so handle that
    is_sorted = check["is_sorted"][0]
    if is_sorted is None:
        is_sorted = True  # Single row is always sorted
    
    if not is_sorted:
        raise LeakageError(
            f"Data is not sorted by {timestamp_col}. "
            "This will cause data leakage in rolling features."
        )
    
    return True


def create_train_test_split(
    df: pl.LazyFrame,
    cutoff_date: date,
    timestamp_col: str = "start_timestamp"
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Create temporal train/test split.
    
    Args:
        df: Full dataset LazyFrame
        cutoff_date: Date to split on (test set starts at this date)
        timestamp_col: Timestamp column name
        
    Returns:
        (train_df, test_df) tuple of LazyFrames
    """
    # Ensure sorted
    df = df.sort(timestamp_col)
    
    # Convert cutoff to timestamp
    cutoff_ts = int(datetime.combine(cutoff_date, datetime.min.time()).timestamp())
    
    train = df.filter(pl.col(timestamp_col) < cutoff_ts)
    test = df.filter(pl.col(timestamp_col) >= cutoff_ts)
    
    # Log split sizes
    train_count = train.select(pl.len()).collect().item()
    test_count = test.select(pl.len()).collect().item()
    
    logger.info(f"Train/Test split at {cutoff_date}")
    logger.info(f"  Train: {train_count:,} matches")
    logger.info(f"  Test:  {test_count:,} matches")
    
    return train, test


def assert_no_leakage(
    train: pl.LazyFrame,
    test: pl.LazyFrame,
    timestamp_col: str = "start_timestamp"
) -> None:
    """
    Assert that no test data appears before train data ends.
    
    Raises:
        LeakageError if leakage detected
    """
    train_max = train.select(pl.col(timestamp_col).max()).collect().item()
    test_min = test.select(pl.col(timestamp_col).min()).collect().item()
    
    if train_max is None or test_min is None:
        return # Cannot check leakage if one set is empty
        
    if test_min <= train_max:
        raise LeakageError(
            f"Data leakage detected! "
            f"Test data starts at {test_min} but train data ends at {train_max}"
        )
    
    logger.info("✓ No temporal leakage detected")


def validate_feature_leakage(
    df: pl.LazyFrame,
    feature_cols: list,
    target_col: str = "player_won"
) -> dict:
    """
    Check for suspiciously high correlations that might indicate leakage.
    
    Args:
        df: Dataset with features
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        Dict of suspicious features with their correlations
    """
    suspicious = {}
    
    # Collect for correlation calculation
    data = df.select([target_col] + feature_cols).collect()
    
    target_values = data[target_col].cast(pl.Float64)
    
    for col in feature_cols:
        if col in data.columns:
            # Check for non-null values
            feature_values = data[col].cast(pl.Float64)
            
            # Simple correlation check
            corr = target_values.pearson_corr(feature_values)
            
            # Flag if correlation > 0.95 (suspiciously perfect)
            if corr is not None and abs(corr) > 0.95:
                suspicious[col] = corr
                logger.warning(f"⚠ Suspicious correlation for {col}: {corr:.4f}")
    
    if not suspicious:
        logger.info("✓ No suspicious feature correlations detected")
    
    return suspicious


def get_temporal_cv_folds(
    df: pl.LazyFrame,
    n_folds: int = 5,
    min_train_size: int = 1000,
    timestamp_col: str = "start_timestamp"
) -> list:
    """
    Create time-series cross-validation folds.
    Each fold uses all previous data for training.
    
    Args:
        df: Full dataset
        n_folds: Number of folds
        min_train_size: Minimum training samples
        timestamp_col: Timestamp column
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Ensure sorted
    df = df.sort(timestamp_col)
    
    # Get unique timestamps
    timestamps = df.select(timestamp_col).collect()[timestamp_col]
    n_samples = len(timestamps)
    
    # Calculate fold boundaries
    fold_size = (n_samples - min_train_size) // n_folds
    
    folds = []
    for i in range(n_folds):
        train_end = min_train_size + (i * fold_size)
        test_start = train_end
        test_end = test_start + fold_size
        
        if i == n_folds - 1:
            test_end = n_samples  # Last fold gets remaining data
        
        folds.append((
            list(range(0, train_end)),
            list(range(test_start, test_end))
        ))
    
    return folds
