"""
Fatigue features - Match load and rest days.
"""
import polars as pl
from .registry import FeatureRegistry


@FeatureRegistry.register("fatigue_features", category="temporal", priority=30)
def add_fatigue_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add fatigue/rest indicators."""
    schema = df.collect_schema().names()
    
    if "match_date" not in schema:
        return df
    
    return df.with_columns([
        (
            pl.col("match_date") - 
            pl.col("match_date").shift(1).over("player_id")
        ).dt.total_days().alias("days_since_last_match")
    ])
