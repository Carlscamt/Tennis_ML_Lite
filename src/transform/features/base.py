"""
Base classes for feature builders.
"""
import polars as pl
from abc import ABC, abstractmethod
from typing import List


class BaseFeatureBuilder(ABC):
    """
    Abstract base class for feature builders.
    
    Subclass this for complex feature logic that benefits from state.
    For simple features, use the @FeatureRegistry.register decorator directly.
    """
    
    name: str = "base_feature"
    category: str = "unknown"
    
    @abstractmethod
    def transform(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply feature transformations."""
        pass
    
    def __call__(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Allow using builder as a function."""
        return self.transform(df)
    
    def get_column_names(self) -> List[str]:
        """Return list of columns this builder creates."""
        return []


def safe_divide(numerator: pl.Expr, denominator: pl.Expr, default: float = 0.0) -> pl.Expr:
    """Safe division with null/zero handling."""
    return pl.when(denominator > 0).then(numerator / denominator).otherwise(default)


def shifted_rolling_mean(
    col: str,
    window: int,
    partition_by: List[str],
    shift: int = 1
) -> pl.Expr:
    """
    Create a shifted rolling mean expression.
    
    Uses shift(1) to ensure current row is not included (temporal safety).
    """
    return (
        pl.col(col)
        .shift(shift)
        .rolling_mean(window_size=window, min_periods=1)
        .over(partition_by)
        .alias(f"{col}_rolling_{window}")
    )


def shifted_rolling_sum(
    col: str,
    window: int,
    partition_by: List[str],
    shift: int = 1
) -> pl.Expr:
    """Create a shifted rolling sum expression."""
    return (
        pl.col(col)
        .shift(shift)
        .rolling_sum(window_size=window, min_periods=1)
        .over(partition_by)
        .alias(f"{col}_rolling_sum_{window}")
    )
