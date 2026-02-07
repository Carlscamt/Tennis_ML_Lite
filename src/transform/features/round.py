"""
Round features - Tournament round encoding.
"""
import polars as pl
from .registry import FeatureRegistry


ROUND_MAP = {
    "Final": 7,
    "Semifinal": 6,
    "Quarterfinal": 5,
    "Round of 16": 4,
    "Round of 32": 3,
    "Round of 64": 2,
    "Round of 128": 1,
    "Qualification": 0,
}


@FeatureRegistry.register("round_features", category="pre_match", priority=10)
def add_round_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """Encode tournament round as numeric value."""
    schema = df.collect_schema().names()
    
    if "round_name" not in schema:
        return df
    
    return df.with_columns([
        pl.col("round_name")
        .replace(ROUND_MAP, default=2)
        .alias("round_num")
    ])
