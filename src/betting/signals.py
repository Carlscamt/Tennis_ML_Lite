"""
Value bet identification and signal generation.
"""
import polars as pl
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValueBetFinder:
    """
    Identifies value betting opportunities with uncertainty filtering.
    
    Filters out bets that have high reported edge but low confidence
    (near decision boundary), which are more likely noise artifacts.
    """
    
    min_edge: float = 0.05        # 5% minimum edge
    min_confidence: float = 0.55  # Minimum model probability
    min_odds: float = 1.20
    max_odds: float = 5.00
    max_bets_per_day: int = 10
    blend_weight: float = 0.5  # 0.5 = 50% Model, 50% Market (Implied)
    
    # Uncertainty thresholds
    min_margin: float = 0.10      # Reject if |p - 0.5| < 0.10
    max_entropy: float = 0.65     # Reject if entropy > 0.65
    use_uncertainty_filter: bool = True
    
    def find_value_bets(self, predictions_df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter predictions to value bets only.
        
        Expects columns: model_prob, odds_player, odds_opponent
        
        Args:
            predictions_df: DataFrame with predictions and odds
            
        Returns:
            Filtered DataFrame with value bets and uncertainty signals
        """
        # Calculate edge and EV
        df = predictions_df.with_columns([
            # Raw implied probability (includes vig)
            (1 / pl.col("odds_player")).alias("implied_prob"),
            (1 / pl.col("odds_opponent")).alias("implied_prob_opp"),
        ])
        
        # Calculate Overround & Fair Market Probability
        df = df.with_columns([
            (pl.col("implied_prob") + pl.col("implied_prob_opp")).alias("overround")
        ])
        
        df = df.with_columns([
            (pl.col("implied_prob") / pl.col("overround")).alias("fair_market_prob")
        ])
        
        # Calculate Blended Probability (Model + Fair Market)
        df = df.with_columns([
            ((pl.col("model_prob") * self.blend_weight) + 
             (pl.col("fair_market_prob") * (1 - self.blend_weight))).alias("blended_prob")
        ])
        
        df = df.with_columns([
            # Edge = blended_prob - implied_prob
            (pl.col("blended_prob") - pl.col("implied_prob")).alias("edge"),
            
            # Expected value per unit
            (
                pl.col("blended_prob") * (pl.col("odds_player") - 1) -
                (1 - pl.col("blended_prob"))
            ).alias("expected_value"),
        ])
        
        # Add uncertainty signals
        df = self._add_uncertainty_signals(df)
        
        # Build filter conditions
        filter_conditions = (
            (pl.col("model_prob") >= self.min_confidence) &
            (pl.col("edge") >= self.min_edge) &
            (pl.col("odds_player") >= self.min_odds) &
            (pl.col("odds_player") <= self.max_odds)
        )
        
        # Add uncertainty filter if enabled
        if self.use_uncertainty_filter:
            filter_conditions = filter_conditions & (
                (pl.col("margin") >= self.min_margin) &
                (pl.col("entropy") <= self.max_entropy)
            )
        
        # Filter to value bets
        value_bets = df.filter(filter_conditions)
        
        # Log rejection stats if uncertainty filter is on
        if self.use_uncertainty_filter:
            rejected_by_uncertainty = len(df.filter(
                (pl.col("model_prob") >= self.min_confidence) &
                (pl.col("edge") >= self.min_edge) &
                (pl.col("odds_player") >= self.min_odds) &
                (pl.col("odds_player") <= self.max_odds) &
                ((pl.col("margin") < self.min_margin) | (pl.col("entropy") > self.max_entropy))
            ))
            if rejected_by_uncertainty > 0:
                logger.info(f"Uncertainty filter rejected {rejected_by_uncertainty} potential bets")
        
        # Sort by edge descending
        value_bets = value_bets.sort("edge", descending=True)
        
        # Limit bets per day
        if len(value_bets) > self.max_bets_per_day:
            value_bets = value_bets.head(self.max_bets_per_day)
        
        logger.info(f"Found {len(value_bets)} value bets from {len(predictions_df)} matches")
        
        return value_bets
    
    def _add_uncertainty_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add margin, entropy, and confidence columns."""
        eps = 1e-10
        
        return df.with_columns([
            # Margin: distance from decision boundary
            pl.col("model_prob").sub(0.5).abs().alias("margin"),
            
            # Entropy: -p*log2(p) - (1-p)*log2(1-p)
            (
                -pl.col("model_prob").clip(eps, 1-eps) * 
                pl.col("model_prob").clip(eps, 1-eps).log(base=2) -
                (1 - pl.col("model_prob").clip(eps, 1-eps)) * 
                (1 - pl.col("model_prob").clip(eps, 1-eps)).log(base=2)
            ).alias("entropy"),
            
            # Confidence: max(p, 1-p)
            pl.when(pl.col("model_prob") > 0.5)
              .then(pl.col("model_prob"))
              .otherwise(1 - pl.col("model_prob"))
              .alias("bet_confidence"),
        ])
    
    def grade_bet(self, edge: float, confidence: float) -> str:
        """
        Assign a grade to a bet opportunity.
        
        Returns:
            Grade string: 'A+', 'A', 'B', 'C', or 'D'
        """
        if edge >= 0.15 and confidence >= 0.65:
            return "A+"
        elif edge >= 0.10 and confidence >= 0.60:
            return "A"
        elif edge >= 0.07 and confidence >= 0.57:
            return "B"
        elif edge >= 0.05 and confidence >= 0.55:
            return "C"
        else:
            return "D"
    
    def add_grades(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add bet grades to DataFrame."""
        grades = []
        for row in df.iter_rows(named=True):
            edge = row.get("edge", 0)
            conf = row.get("model_prob", 0)
            grades.append(self.grade_bet(edge, conf))
        
        return df.with_columns([
            pl.Series("bet_grade", grades)
        ])
    
    def format_for_display(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Format value bets for dashboard display.
        
        Selects and renames columns for readability.
        """
        display_cols = []
        
        # Core info
        if "player_name" in df.columns:
            display_cols.append(pl.col("player_name").alias("Player"))
        if "opponent_name" in df.columns:
            display_cols.append(pl.col("opponent_name").alias("Opponent"))
        if "tournament_name" in df.columns:
            display_cols.append(pl.col("tournament_name").alias("Tournament"))
        
        # Betting info
        display_cols.extend([
            (pl.col("model_prob") * 100).round(1).alias("Confidence (%)"),
            pl.col("odds_player").round(2).alias("Odds"),
            (pl.col("edge") * 100).round(1).alias("Edge (%)"),
            (pl.col("expected_value") * 100).round(1).alias("EV (%)"),
        ])
        
        # Grade if available
        if "bet_grade" in df.columns:
            display_cols.append(pl.col("bet_grade").alias("Grade"))
        
        # Stake if available
        if "stake_pct" in df.columns:
            display_cols.append((pl.col("stake_pct") * 100).round(1).alias("Stake (%)"))
        
        return df.select(display_cols)


def calculate_ev_confidence_matrix(
    predictions_df: pl.DataFrame,
    edge_thresholds: List[float] = [0.03, 0.05, 0.07, 0.10],
    conf_thresholds: List[float] = [0.52, 0.55, 0.60, 0.65]
) -> pl.DataFrame:
    """
    Calculate ROI for different edge/confidence combinations.
    Useful for threshold optimization.
    
    Args:
        predictions_df: Historical predictions with outcomes
        edge_thresholds: Edge thresholds to test
        conf_thresholds: Confidence thresholds to test
        
    Returns:
        DataFrame with ROI for each combination
    """
    results = []
    
    for edge_min in edge_thresholds:
        for conf_min in conf_thresholds:
            # Filter
            filtered = predictions_df.filter(
                (pl.col("edge") >= edge_min) &
                (pl.col("model_prob") >= conf_min)
            )
            
            if len(filtered) == 0:
                continue
            
            # Calculate ROI
            total_bets = len(filtered)
            wins = filtered.filter(pl.col("player_won")).select(pl.len()).item()
            
            # Flat betting ROI
            profit = filtered.select([
                pl.when(pl.col("player_won"))
                .then(pl.col("odds_player") - 1)
                .otherwise(-1)
                .sum()
                .alias("profit")
            ]).item()
            
            roi = profit / total_bets if total_bets > 0 else 0
            
            results.append({
                "min_edge": edge_min,
                "min_conf": conf_min,
                "total_bets": total_bets,
                "wins": wins,
                "win_rate": wins / total_bets,
                "profit": profit,
                "roi": roi,
            })
    
    return pl.DataFrame(results)
