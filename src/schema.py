"""
Unified Data Schema for Tennis Betting ML Pipeline with Pandera.
"""
from typing import Optional, List, Dict
import pandera.polars as pa
from pandera import Column, Check, Field
import polars as pl
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

class TennisMatchesSchema(pa.DataFrameModel):
    """
    Schema for incoming tennis match data (Raw).
    """
    # Match identifiers
    event_id: int = Field(unique=False, description="Event ID (can repeat per match for diff players)")
    player_id: int = Field(description="Player ID")
    opponent_id: int = Field(description="Opponent ID")
    
    start_timestamp: int = Field(ge=0, description="Unix timestamp")
    
    # Player info
    player_name: str = Field()
    opponent_name: str = Field()
    
    # Metadata
    tournament_name: str = Field(nullable=True)
    ground_type: str = Field(nullable=True)
    
    # Betting odds (Critical) - Allow nulls for historical, but strict for predictions if needed
    odds_player: float = Field(gt=1.0, nullable=True, description="Decimal odds for Player")
    odds_opponent: float = Field(gt=1.0, nullable=True, description="Decimal odds for Opponent")
    
    # Outcome (Optional/Nullable for upcoming)
    player_won: bool = Field(nullable=True)

    class Config:
        # We can add DataFrame level checks here
        pass

class FeaturesSchema(pa.DataFrameModel):
    """
    Schema for engineered features used in training/prediction.
    """
    event_id: int
    player_id: int
    
    # Critical Features requiring validation
    odds_player: float = Field(gt=1.0, nullable=True)
    implied_prob_player: float = Field(ge=0.0, le=1.0, nullable=True)
    
    player_win_rate_20: float = Field(ge=0.0, le=1.0, nullable=True)
    
    class Config:
        pass

# =============================================================================
# SCHEMA VALIDATOR
# =============================================================================

class SchemaValidator:
    """
    Unified schema validation for tennis data using Pandera.
    """
    
    def __init__(self):
        self.raw_schema = TennisMatchesSchema
        self.features_schema = FeaturesSchema
    
    def validate_raw_data(self, df: pl.DataFrame) -> dict:
        """
        Validate raw incoming data against TennisMatchesSchema.
        """
        try:
            # Clean dataframe for validation if needed (e.g. strict types)
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            
            validated = self.raw_schema.validate(df, lazy=True)
            
            # Count rows safely
            if isinstance(df, pl.LazyFrame):
                num_rows = df.select(pl.len()).collect().item()
            else:
                num_rows = len(df)
            
            return {
                'valid': True,
                'num_rows': num_rows,
                'num_invalid_rows': 0,
                'errors': [],
                'error_counts': {},
            }
        
        except pa.errors.SchemaError as e:
            # Single failure
            msg = str(e)
            return {
                'valid': False,
                'num_rows': len(df),
                'num_invalid_rows': len(df),
                'errors': [msg],
                'error_counts': {'schema_error': 1},
            }
        except pa.errors.SchemaErrors as e:
            # Lazy validation aggregation
            errors = e.failure_cases.to_dicts()
            error_msgs = [f"{err['column']}: {err['check']}" for err in errors]
            return {
                'valid': False,
                'num_rows': len(df),
                'num_invalid_rows': len(e.failure_cases),
                'errors': error_msgs,
                'error_counts': {'multi_schema_error': len(errors)},
            }
            
    def validate_features(self, df: pl.DataFrame) -> dict:
        """Validate engineered features against FeaturesSchema."""
        try:
            self.features_schema.validate(df, lazy=True)
            return {
                'valid': True,
                'num_rows': len(df),
                'num_invalid_rows': 0,
                'errors': [],
            }
        except pa.errors.SchemaErrors as e:
            errors = e.failure_cases.to_dicts()
            return {
                'valid': False,
                'num_rows': len(df),
                'num_invalid_rows': len(e.failure_cases),
                'errors': [str(x) for x in errors],
            }

# =============================================================================
# LEGACY HELPERS (Backwards compatibility)
# =============================================================================

def merge_datasets(existing: pl.DataFrame, new_data: pl.DataFrame, prefer_new: bool = True) -> pl.DataFrame:
    """Merge logic (legacy wrapper, implementing robust concatenation)."""
    try:
        combined = pl.concat([existing, new_data], how="diagonal_relaxed")
        combined = combined.unique(subset=["event_id", "player_id"], keep="last" if prefer_new else "first")
        return combined.sort("start_timestamp")
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return existing
