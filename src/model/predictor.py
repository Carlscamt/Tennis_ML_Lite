"""
Prediction service for live and batch predictions.
"""
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Predictor:
    """
    Prediction service for tennis match outcomes.
    Handles both live predictions and batch processing.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Args:
            model_path: Path to saved model (loads on init if provided)
        """
        self.model = None
        self.trainer = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, path: Path) -> None:
        """Load a trained model."""
        from .trainer import ModelTrainer
        
        self.trainer = ModelTrainer()
        self.trainer.load(path)
        self.model = self.trainer.model
        
        logger.info(f"Loaded model with {len(self.trainer.feature_columns)} features")
    
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add predictions to dataframe.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            DataFrame with prediction columns added
        """
        if self.trainer is None:
            raise ValueError("No model loaded")
        
        probas = self.trainer.predict_proba(df)
        
        return df.with_columns([
            pl.Series("model_prob", probas),
            pl.Series("model_prediction", (probas >= 0.5).astype(int)),
        ])
    
    def predict_with_value(
        self,
        df: pl.DataFrame,
        min_edge: float = 0.05
    ) -> pl.DataFrame:
        """
        Add predictions with betting value calculations.
        
        Args:
            df: DataFrame with odds_player column
            min_edge: Minimum edge to flag as value bet
            
        Returns:
            DataFrame with model_prob, edge, and is_value_bet columns
        """
        df = self.predict(df)
        
        # Calculate edge if odds available
        if "odds_player" in df.columns:
            df = df.with_columns([
                # Implied probability from odds
                (1 / pl.col("odds_player")).alias("implied_prob"),
                
                # Edge = model_prob - implied_prob
                (pl.col("model_prob") - (1 / pl.col("odds_player"))).alias("edge"),
            ])
            
            df = df.with_columns([
                # Flag value bets
                (pl.col("edge") >= min_edge).alias("is_value_bet"),
                
                # Expected value per unit bet
                (
                    pl.col("model_prob") * (pl.col("odds_player") - 1) -
                    (1 - pl.col("model_prob"))
                ).alias("expected_value"),
            ])
        
        return df
    
    def get_todays_predictions(
        self,
        matches_df: pl.DataFrame,
        min_confidence: float = 0.55,
        min_edge: float = 0.05
    ) -> pl.DataFrame:
        """
        Get today's betting recommendations.
        
        Args:
            matches_df: Upcoming matches with features and odds
            min_confidence: Minimum model probability
            min_edge: Minimum edge for value bets
            
        Returns:
            DataFrame with recommended bets
        """
        predictions = self.predict_with_value(matches_df, min_edge)
        
        # Filter to value bets
        value_bets = predictions.filter(
            (pl.col("model_prob") >= min_confidence) &
            (pl.col("is_value_bet") == True)
        )
        
        # Sort by edge
        value_bets = value_bets.sort("edge", descending=True)
        
        # Add timestamp
        value_bets = value_bets.with_columns([
            pl.lit(datetime.now().isoformat()).alias("prediction_timestamp")
        ])
        
        logger.info(f"Found {len(value_bets)} value bets from {len(predictions)} matches")
        
        return value_bets
    
    def save_predictions(
        self,
        predictions: pl.DataFrame,
        output_dir: Path,
        prefix: str = "predictions"
    ) -> Path:
        """
        Save predictions to parquet file.
        
        Args:
            predictions: Predictions DataFrame
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.parquet"
        path = output_dir / filename
        
        predictions.write_parquet(path)
        logger.info(f"Saved predictions to {path}")
        
        return path
