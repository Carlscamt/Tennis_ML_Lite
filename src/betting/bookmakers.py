"""
Multi-bookmaker odds selection and line shopping strategies.

Provides consistent bookmaker selection rules for optimal odds.
"""
import polars as pl
import numpy as np
from typing import List, Optional, Literal
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# Vetted bookmakers (reliable, liquid markets)
TIER1_BOOKS = [
    "pinnacle", "bet365", "unibet", "betfair", 
    "williamhill", "bwin", "marathonbet"
]

TIER2_BOOKS = [
    "betway", "888sport", "betclic", "interwetten",
    "betsson", "nordicbet", "betfred"
]


@dataclass
class BookmakerConfig:
    """Configuration for bookmaker selection."""
    
    strategy: Literal["max", "percentile", "average", "single"] = "max"
    percentile: float = 75.0  # For percentile strategy (avoid outliers)
    single_book: Optional[str] = "pinnacle"  # For single strategy
    
    # Book filtering
    use_tier1_only: bool = False
    allowed_books: Optional[List[str]] = None  # Override default vetted list
    excluded_books: List[str] = field(default_factory=list)
    
    # Validation
    min_books_required: int = 2  # Min books for max/percentile
    

class BookmakerSelector:
    """
    Selects best odds from multiple bookmakers.
    
    Strategies:
    - max: Best available odds from vetted books
    - percentile: Nth percentile to avoid outliers
    - average: Average across all books
    - single: Use specific bookmaker
    """
    
    def __init__(self, config: Optional[BookmakerConfig] = None):
        self.config = config or BookmakerConfig()
        self._vetted_books = self._get_vetted_books()
    
    def _get_vetted_books(self) -> List[str]:
        """Get list of vetted bookmakers."""
        if self.config.allowed_books:
            books = self.config.allowed_books
        elif self.config.use_tier1_only:
            books = TIER1_BOOKS.copy()
        else:
            books = TIER1_BOOKS + TIER2_BOOKS
        
        # Remove excluded
        return [b for b in books if b not in self.config.excluded_books]
    
    def select_odds(
        self,
        odds_dict: dict,
        player_key: str = "home"
    ) -> Optional[float]:
        """
        Select odds based on configured strategy.
        
        Args:
            odds_dict: Dict of {bookmaker: {home: odds, away: odds}}
            player_key: "home" or "away"
            
        Returns:
            Selected odds value
        """
        if not odds_dict:
            return None
        
        # Extract odds from vetted books
        odds_values = []
        for book, odds in odds_dict.items():
            book_lower = book.lower()
            if book_lower in self._vetted_books or not self._vetted_books:
                if isinstance(odds, dict) and player_key in odds:
                    val = odds[player_key]
                elif isinstance(odds, (int, float)):
                    val = odds
                else:
                    continue
                    
                if val and val > 1.0:
                    odds_values.append(val)
        
        if not odds_values:
            return None
        
        return self._apply_strategy(odds_values)
    
    def select_odds_from_columns(
        self,
        df: pl.DataFrame,
        book_columns: List[str],
        output_col: str = "selected_odds"
    ) -> pl.DataFrame:
        """
        Select odds from multiple bookmaker columns.
        
        Args:
            df: DataFrame with bookmaker odds columns
            book_columns: List of column names (e.g., ["odds_pinnacle", "odds_bet365"])
            output_col: Name of output column
            
        Returns:
            DataFrame with selected odds column
        """
        # Filter to columns that exist
        existing_cols = [c for c in book_columns if c in df.columns]
        
        if not existing_cols:
            logger.warning("No bookmaker columns found")
            return df.with_columns(pl.lit(None).alias(output_col))
        
        if self.config.strategy == "max":
            return df.with_columns([
                pl.max_horizontal(*existing_cols).alias(output_col)
            ])
        
        elif self.config.strategy == "average":
            # Average of non-null values
            return df.with_columns([
                pl.mean_horizontal(*existing_cols).alias(output_col)
            ])
        
        elif self.config.strategy == "percentile":
            # Calculate percentile for each row
            def row_percentile(row):
                vals = [row[c] for c in existing_cols if row[c] is not None]
                if not vals:
                    return None
                return np.percentile(vals, self.config.percentile)
            
            # Use apply for percentile
            result = df.with_columns([
                pl.struct(existing_cols).map_elements(
                    lambda s: self._percentile_odds([s[c] for c in existing_cols]),
                    return_dtype=pl.Float64
                ).alias(output_col)
            ])
            return result
        
        elif self.config.strategy == "single":
            # Use single book column
            book_col = f"odds_{self.config.single_book}" if self.config.single_book else existing_cols[0]
            if book_col in df.columns:
                return df.with_columns([
                    pl.col(book_col).alias(output_col)
                ])
            else:
                logger.warning(f"Book column {book_col} not found, using first available")
                return df.with_columns([
                    pl.col(existing_cols[0]).alias(output_col)
                ])
        
        return df
    
    def _apply_strategy(self, odds_values: List[float]) -> float:
        """Apply selection strategy to list of odds."""
        if not odds_values:
            return 0.0
        
        if self.config.strategy == "max":
            return max(odds_values)
        
        elif self.config.strategy == "percentile":
            return self._percentile_odds(odds_values)
        
        elif self.config.strategy == "average":
            return sum(odds_values) / len(odds_values)
        
        elif self.config.strategy == "single":
            return odds_values[0]  # First available
        
        return max(odds_values)
    
    def _percentile_odds(self, odds_values: List[Optional[float]]) -> Optional[float]:
        """Calculate percentile of odds, filtering None values."""
        vals = [v for v in odds_values if v is not None and v > 1.0]
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]
        return float(np.percentile(vals, self.config.percentile))
    
    def compare_strategies(
        self,
        df: pl.DataFrame,
        book_columns: List[str]
    ) -> pl.DataFrame:
        """
        Compare all strategies for analysis.
        
        Returns DataFrame with odds columns for each strategy.
        """
        existing_cols = [c for c in book_columns if c in df.columns]
        
        if not existing_cols:
            return df
        
        # Add all strategy columns
        df = df.with_columns([
            pl.max_horizontal(*existing_cols).alias("odds_max"),
            pl.mean_horizontal(*existing_cols).alias("odds_avg"),
        ])
        
        # First book as "single"
        if existing_cols:
            df = df.with_columns([
                pl.col(existing_cols[0]).alias("odds_single")
            ])
        
        # Percentile (75th)
        df = df.with_columns([
            pl.struct(existing_cols).map_elements(
                lambda s: self._percentile_odds([s[c] for c in existing_cols]),
                return_dtype=pl.Float64
            ).alias("odds_p75")
        ])
        
        return df


def analyze_line_shopping_impact(
    df: pl.DataFrame,
    book_columns: List[str],
    outcome_col: str = "player_won"
) -> dict:
    """
    Analyze ROI impact of different book selection strategies.
    
    Returns dict with ROI comparison.
    """
    selector = BookmakerSelector()
    compared = selector.compare_strategies(df, book_columns)
    
    results = {}
    for strategy in ["max", "avg", "single", "p75"]:
        odds_col = f"odds_{strategy}"
        if odds_col not in compared.columns:
            continue
        
        valid = compared.filter(pl.col(odds_col).is_not_null())
        
        if len(valid) == 0:
            continue
        
        # Calculate ROI
        profit = valid.with_columns([
            (
                pl.col(outcome_col).cast(pl.Int32) * (pl.col(odds_col) - 1) -
                (1 - pl.col(outcome_col).cast(pl.Int32))
            ).alias("profit")
        ])
        
        total_profit = profit["profit"].sum()
        total_bets = len(profit)
        roi = total_profit / total_bets * 100 if total_bets else 0
        
        results[strategy] = {
            "bets": total_bets,
            "avg_odds": valid[odds_col].mean(),
            "total_profit": total_profit,
            "roi_pct": roi
        }
    
    return results
