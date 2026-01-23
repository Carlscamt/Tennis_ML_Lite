"""
Unified Data Schema for Tennis Betting ML Pipeline.

Defines the canonical schema for all data flowing through the pipeline.
All scrapers and processors should validate against this schema.
"""
import polars as pl
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CORE SCHEMA DEFINITION
# =============================================================================

# Required columns for all match data
CORE_COLUMNS = {
    "event_id": pl.Int64,
    "player_id": pl.Int64,
    "opponent_id": pl.Int64,
    "start_timestamp": pl.Int64,
}

# Match outcome columns (null for upcoming matches)
OUTCOME_COLUMNS = {
    "player_won": pl.Boolean,
    "player_sets": pl.Int64,
    "opponent_sets": pl.Int64,
}

# Match metadata (pre-match known)
METADATA_COLUMNS = {
    "player_name": pl.String,
    "opponent_name": pl.String,
    "tournament_id": pl.Int64,
    "tournament_name": pl.String,
    "round_name": pl.String,
    "ground_type": pl.String,
}

# Odds columns (pre-match known)
ODDS_COLUMNS = {
    "odds_player": pl.Float64,
    "odds_opponent": pl.Float64,
}

# Scraper metadata
INTERNAL_COLUMNS = {
    "_schema_version": pl.String,
    "_scraped_at": pl.String,
    "_data_type": pl.String,  # "historical" or "upcoming"
}

# All columns combined
FULL_SCHEMA = {
    **CORE_COLUMNS,
    **OUTCOME_COLUMNS,
    **METADATA_COLUMNS,
    **ODDS_COLUMNS,
    **INTERNAL_COLUMNS,
}

# Current schema version
SCHEMA_VERSION = "2.0"


# =============================================================================
# SCHEMA VALIDATOR
# =============================================================================

@dataclass
class ValidationResult:
    """Result of schema validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]


class SchemaValidator:
    """Validates DataFrames against the canonical schema."""
    
    def __init__(self, strict: bool = False):
        """
        Args:
            strict: If True, fail on any missing required column
        """
        self.strict = strict
    
    def validate(self, df: pl.DataFrame, data_type: str = "historical") -> ValidationResult:
        """
        Validate a DataFrame against the schema.
        
        Args:
            df: DataFrame to validate
            data_type: "historical" or "upcoming"
            
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check required core columns
        for col in CORE_COLUMNS:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check outcome columns (only required for historical)
        if data_type == "historical":
            for col in OUTCOME_COLUMNS:
                if col not in df.columns:
                    warnings.append(f"Missing outcome column: {col}")
        
        # Check for duplicates
        if "event_id" in df.columns and "player_id" in df.columns:
            n_dupes = len(df) - len(df.unique(["event_id", "player_id"]))
            if n_dupes > 0:
                errors.append(f"Found {n_dupes} duplicate event_id+player_id rows")
        
        # Check odds columns
        if "odds_player" in df.columns:
            null_odds = df.filter(pl.col("odds_player").is_null()).height
            odds_coverage = 1 - (null_odds / len(df)) if len(df) > 0 else 0
            if odds_coverage < 0.01:
                warnings.append(f"Very low odds coverage: {odds_coverage:.1%}")
        
        # Check temporal ordering
        if "start_timestamp" in df.columns:
            is_sorted = df["start_timestamp"].is_sorted()
            if not is_sorted:
                warnings.append("Data is not sorted by start_timestamp")
        
        valid = len(errors) == 0 if self.strict else True
        
        return ValidationResult(valid=valid, errors=errors, warnings=warnings)
    
    def validate_and_log(self, df: pl.DataFrame, data_type: str = "historical") -> bool:
        """Validate and log results."""
        result = self.validate(df, data_type)
        
        for error in result.errors:
            logger.error(f"[SCHEMA ERROR] {error}")
        
        for warning in result.warnings:
            logger.warning(f"[SCHEMA WARN] {warning}")
        
        if result.valid:
            logger.info(f"[SCHEMA] Validation passed ({len(df)} rows)")
        
        return result.valid


# =============================================================================
# DEDUPLICATION UTILITIES
# =============================================================================

def deduplicate_matches(
    df: pl.LazyFrame,
    prefer_with_odds: bool = True
) -> pl.LazyFrame:
    """
    Deduplicate matches, keeping one row per event_id + player_id.
    
    Args:
        df: LazyFrame with match data
        prefer_with_odds: If True, prefer rows with odds data
        
    Returns:
        Deduplicated LazyFrame
    """
    if prefer_with_odds:
        # Add priority column: rows with odds get priority
        df = df.with_columns([
            pl.col("odds_player").is_not_null().cast(pl.Int8).alias("_has_odds")
        ])
        
        # Sort by timestamp (chronological), then odds priority (descending)
        df = df.sort(["start_timestamp", "_has_odds"], descending=[False, True])
        
        # Keep first (which has odds if available)
        df = df.unique(subset=["event_id", "player_id"], keep="first", maintain_order=True)
        
        # Drop helper column
        df = df.drop("_has_odds")
    else:
        # Simple dedup, keep first by timestamp
        df = df.sort("start_timestamp")
        df = df.unique(subset=["event_id", "player_id"], keep="first", maintain_order=True)
    
    return df


def merge_datasets(
    existing: pl.DataFrame,
    new_data: pl.DataFrame,
    prefer_new: bool = True
) -> pl.DataFrame:
    """
    Merge new data into existing dataset, handling duplicates.
    
    Args:
        existing: Existing dataset
        new_data: New data to merge
        prefer_new: If True, new data overrides existing
        
    Returns:
        Merged DataFrame
    """
    # Concatenate
    combined = pl.concat([existing, new_data], how="diagonal")
    
    # Add source priority
    combined = combined.with_columns([
        pl.when(pl.col("_scraped_at") == new_data["_scraped_at"].max())
        .then(pl.lit(1 if prefer_new else 0))
        .otherwise(pl.lit(0 if prefer_new else 1))
        .alias("_priority")
    ])
    
    # Also prefer rows with odds
    combined = combined.with_columns([
        pl.col("odds_player").is_not_null().cast(pl.Int8).alias("_has_odds")
    ])
    
    # Sort by priority, then odds
    combined = combined.sort(
        ["event_id", "player_id", "_priority", "_has_odds"],
        descending=[False, False, True, True]
    )
    
    # Keep first per event+player
    result = combined.unique(
        subset=["event_id", "player_id"],
        keep="first",
        maintain_order=True
    )
    
    # Clean up
    result = result.drop(["_priority", "_has_odds"])
    result = result.sort("start_timestamp")
    
    return result


# =============================================================================
# SCHEMA ENFORCEMENT
# =============================================================================

def enforce_schema(
    df: pl.DataFrame,
    add_missing: bool = True,
    data_type: str = "historical"
) -> pl.DataFrame:
    """
    Enforce the canonical schema on a DataFrame.
    
    Args:
        df: DataFrame to process
        add_missing: Add missing optional columns with null values
        data_type: "historical" or "upcoming"
        
    Returns:
        DataFrame with enforced schema
    """
    # Add schema version
    if "_schema_version" not in df.columns:
        df = df.with_columns(pl.lit(SCHEMA_VERSION).alias("_schema_version"))
    
    # Add data type
    if "_data_type" not in df.columns:
        df = df.with_columns(pl.lit(data_type).alias("_data_type"))
    
    # Cast core columns to correct types
    for col, dtype in CORE_COLUMNS.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
    
    # For odds, ensure float type
    for col in ODDS_COLUMNS:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
    
    return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def prepare_for_ml(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Prepare data for ML pipeline.
    Combines deduplication and schema enforcement.
    """
    # Deduplicate
    df = deduplicate_matches(df, prefer_with_odds=True)
    
    # Sort chronologically
    df = df.sort("start_timestamp")
    
    return df


def get_safe_feature_patterns() -> List[str]:
    """Return patterns for features that are safe to use (no leakage)."""
    return [
        "win_rate_",           # Rolling win rates (shifted)
        "h2h_",                # Head-to-head (shifted)
        "surface_win_rate",    # Surface-specific (shifted)
        "days_since",          # Pre-match known
        "odds_",               # Pre-match known
        "implied_prob",        # Derived from pre-match odds
        "odds_ratio",          # Derived from pre-match odds
        "is_underdog",         # Derived from pre-match odds
        "round_num",           # Pre-match known
        "_avg_",               # Shifted rolling averages
        "_pct_avg_",           # Shifted rolling percentages
        "_ratio_avg_",         # Shifted rolling ratios
    ]
