"""
Polars utility functions.
"""
import polars as pl
from typing import Optional


def safe_divide(
    numerator: pl.Expr,
    denominator: pl.Expr,
    default: float = 0.0
) -> pl.Expr:
    """
    Safe division that handles zero denominators.
    
    Args:
        numerator: Numerator expression
        denominator: Denominator expression
        default: Value to use when denominator is zero
        
    Returns:
        Polars expression with safe division
    """
    return (
        pl.when(denominator != 0)
        .then(numerator / denominator)
        .otherwise(default)
    )


def fillna_grouped(
    column: str,
    group_by: str,
    method: str = "mean"
) -> pl.Expr:
    """
    Fill nulls with grouped statistic.
    
    Args:
        column: Column to fill
        group_by: Column to group by
        method: 'mean', 'median', or 'ffill'
        
    Returns:
        Polars expression
    """
    if method == "mean":
        fill_value = pl.col(column).mean().over(group_by)
    elif method == "median":
        fill_value = pl.col(column).median().over(group_by)
    elif method == "ffill":
        fill_value = pl.col(column).forward_fill().over(group_by)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return pl.col(column).fill_null(fill_value)


def add_rolling_stats(
    df: pl.LazyFrame,
    column: str,
    group_by: str,
    windows: list,
    stats: list = ["mean", "std"]
) -> pl.LazyFrame:
    """
    Add multiple rolling statistics for a column.
    
    Args:
        df: Input LazyFrame
        column: Column to compute stats for
        group_by: Column to partition by
        windows: List of window sizes
        stats: Statistics to compute ('mean', 'std', 'min', 'max')
        
    Returns:
        LazyFrame with new columns
    """
    new_cols = []
    
    for window in windows:
        for stat in stats:
            if stat == "mean":
                expr = (
                    pl.col(column)
                    .shift(1)
                    .rolling_mean(window_size=window, min_periods=1)
                    .over(group_by)
                    .alias(f"{column}_roll_{stat}_{window}")
                )
            elif stat == "std":
                expr = (
                    pl.col(column)
                    .shift(1)
                    .rolling_std(window_size=window, min_periods=2)
                    .over(group_by)
                    .alias(f"{column}_roll_{stat}_{window}")
                )
            elif stat == "min":
                expr = (
                    pl.col(column)
                    .shift(1)
                    .rolling_min(window_size=window, min_periods=1)
                    .over(group_by)
                    .alias(f"{column}_roll_{stat}_{window}")
                )
            elif stat == "max":
                expr = (
                    pl.col(column)
                    .shift(1)
                    .rolling_max(window_size=window, min_periods=1)
                    .over(group_by)
                    .alias(f"{column}_roll_{stat}_{window}")
                )
            else:
                continue
            
            new_cols.append(expr)
    
    if new_cols:
        df = df.with_columns(new_cols)
    
    return df
