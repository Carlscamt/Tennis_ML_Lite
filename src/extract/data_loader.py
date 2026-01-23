"""
Data loading utilities using Polars.
Loads raw scraped data and prepares it for feature engineering.
"""
import polars as pl
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def load_raw_matches(path: Path) -> pl.LazyFrame:
    """
    Load a single parquet file as a LazyFrame.
    
    Args:
        path: Path to parquet file
        
    Returns:
        pl.LazyFrame with raw match data
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    return pl.scan_parquet(path)


def load_all_parquet_files(directory: Path, pattern: str = "*.parquet") -> pl.LazyFrame:
    """
    Load and concatenate all parquet files from a directory.
    
    Args:
        directory: Directory containing parquet files
        pattern: Glob pattern for files
        
    Returns:
        pl.LazyFrame with all matches concatenated
    """
    files = list(directory.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No parquet files found in {directory}")
    
    logger.info(f"Loading {len(files)} parquet files from {directory}")
    
    # Use diagonal concat to handle schema differences
    dfs = [pl.scan_parquet(f) for f in files]
    return pl.concat(dfs, how="diagonal")


def validate_required_columns(df: pl.LazyFrame, required: List[str]) -> bool:
    """
    Check if LazyFrame has all required columns.
    
    Args:
        df: LazyFrame to check
        required: List of required column names
        
    Returns:
        True if all columns present
        
    Raises:
        ValueError if columns are missing
    """
    schema = df.collect_schema()
    missing = [col for col in required if col not in schema.names()]
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return True


def prepare_base_dataset(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Prepare base dataset with standardized columns and types.
    
    Args:
        df: Raw LazyFrame
        
    Returns:
        Cleaned LazyFrame ready for feature engineering
    """
    # Required columns for ML
    required = [
        "event_id",
        "player_id",
        "opponent_id",
        "player_won",
        "start_timestamp",
    ]
    
    validate_required_columns(df, required)
    
    return df.with_columns([
        # Convert timestamp to datetime
        pl.from_epoch("start_timestamp").alias("match_date"),
        
        # Ensure boolean type for target
        pl.col("player_won").cast(pl.Boolean),
        
        # Ensure integer IDs
        pl.col("player_id").cast(pl.Int64),
        pl.col("opponent_id").cast(pl.Int64),
        pl.col("event_id").cast(pl.Int64),
    ]).sort("start_timestamp")


def get_dataset_stats(df: pl.LazyFrame) -> dict:
    """
    Get summary statistics for a dataset.
    
    Returns:
        Dict with counts, date range, coverage stats
    """
    stats = df.select([
        pl.len().alias("total_matches"),
        pl.col("player_id").n_unique().alias("unique_players"),
        pl.col("event_id").n_unique().alias("unique_events"),
        pl.col("match_date").min().alias("earliest_match"),
        pl.col("match_date").max().alias("latest_match"),
    ]).collect().to_dicts()[0]
    
    # Odds coverage if available
    if "odds_player" in df.collect_schema().names():
        odds_stats = df.select([
            pl.col("odds_player").is_not_null().sum().alias("with_odds"),
        ]).collect().to_dicts()[0]
        stats["odds_coverage"] = odds_stats["with_odds"] / stats["total_matches"]
    
    return stats
