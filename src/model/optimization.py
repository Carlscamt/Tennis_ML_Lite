"""
Hyperparameter optimization with Profit (ROI) as the objective.
"""
import logging
import numpy as np
import polars as pl
from typing import Dict, List, Any, Optional, Callable
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
from src.model.trainer import ModelTrainer

logger = logging.getLogger(__name__)

class ProfitOptimizer:
    """
    Optimizes model hyperparameters by maximizing betting ROI 
    in a Walk-Forward Validation loop.
    """
    
    def __init__(self, feature_cols: List[str], target_col: str = "player_won", n_splits: int = 3):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_splits = n_splits
        self.best_params = None
        self.best_roi = -float("inf")
        self.results = []

    def optimize(self, train_df: pl.DataFrame, param_grid: Dict[str, List[Any]], top_k: int = 1) -> Dict[str, Any]:
        """
        Run grid/random search to find best parameters.
        
        Args:
            train_df: DataFrame containing features, target, AND odds columns
                      (odds_player, odds_opponent)
            param_grid: Dictionary of parameters to try
            top_k: Number of top results to keep
            
        Returns:
            Best parameter set
        """
        # Generate all combinations (Grid Search)
        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        logger.info(f"Starting Profit Optimization: {len(param_combinations)} combinations, {self.n_splits}-fold TimeSeries CV")
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # Sort by date just in case
        if "match_date" in train_df.columns:
            train_df = train_df.sort("match_date")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing params [{i+1}/{len(param_combinations)}]: {params}")
            fold_rois = []
            
            # Walk-Forward CV
            # We need to manually index because Polars doesn't support sklearn split directly on DF
            # So we split indices
            indices = np.arange(len(train_df))
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(indices)):
                # Split Data
                fold_train = train_df[train_idx]
                fold_val = train_df[val_idx]
                
                # Train Model
                trainer = ModelTrainer(params=params, calibrate=False)  # Skip calibration for speed during search? 
                                                                       # Or keep True? Usually True is better for value betting.
                                                                       # Let's default to False for raw speed, or make it configurable. 
                                                                       # Actually, uncalibrated probabilities might hurt value betting.
                                                                       # I'll enable calibration if dataset is large enough, but for now False.
                
                # We need to suppress logs during training loop
                trainer.train(fold_train, self.feature_cols, self.target_col)
                
                # Predict
                preds = trainer.predict_proba(fold_val)
                
                # Evaluate ROI (Mini-Backtest)
                roi = self._calculate_roi(fold_val, preds)
                fold_rois.append(roi)
                
            avg_roi = np.mean(fold_rois)
            logger.info(f"  -> Avg ROI: {avg_roi:.2f}% (Folds: {[f'{r:.1f}' for r in fold_rois]})")
            
            self.results.append({
                "params": params,
                "roi": avg_roi,
                "fold_scores": fold_rois
            })
            
            if avg_roi > self.best_roi:
                self.best_roi = avg_roi
                self.best_params = params
                
        logger.info(f"Optimization Complete. Best ROI: {self.best_roi:.2f}%")
        return self.best_params

    def _calculate_roi(self, df: pl.DataFrame, probas: np.ndarray) -> float:
        """
        Simple flat-stake betting strategy for evaluation.
        Bet if Edge > 0.
        """
        # Add probas to DF
        df = df.with_columns(pl.Series("prob", probas))
        
        # Simple Value Bet Logic (mimics ValueBetFinder but simplified for speed)
        # Edge = Prob - (1/Odds)
        # If Edge > 0, Bet
        
        # We need check both Home and Away? 
        # Usually model predicts Home Win.
        
        # Filter for Home Bets
        # Edge = Prob - Implied
        bets = df.filter(
            (pl.col("prob") > (1 / pl.col("odds_player"))) 
        )
        
        if len(bets) == 0:
            return 0.0
            
        # Calculate Profit
        # Profit = Expected_Return - Stake
        # If Won: (Odds - 1)
        # If Lost: -1
        
        # We assume flat stake of 1 unit
        # Win = (Odds - 1) * 1
        # Loss = -1
        
        winnings = bets.filter(pl.col(self.target_col) == 1).select(
            (pl.col("odds_player") - 1).sum()
        ).item()
        
        losses = bets.filter(pl.col(self.target_col) == 0).height
        
        profit = winnings - losses
        total_staked = len(bets)
        
        return (profit / total_staked) * 100 if total_staked > 0 else 0.0
