"""
Unit tests for Profit Optimization Logic.
"""
import pytest
import polars as pl
import numpy as np
from src.model.optimization import ProfitOptimizer

class TestProfitOptimizer:
    
    def test_calculate_roi_wins_and_losses(self):
        """
        Test ROI calc:
        1. Bet Won @ 2.0 -> Profit +1
        2. Bet Lost @ 2.0 -> Profit -1
        Total Staked: 2
        Total Profit: 0
        ROI: 0%
        """
        optimizer = ProfitOptimizer(feature_cols=["f1"])
        
        # DataFrame with features and odds (dummy features)
        df = pl.DataFrame({
            "f1": [1, 1],
            "odds_player": [2.0, 2.0],
            "player_won": [1, 0]  # 1 Win, 1 Loss
        })
        
        # Probas: High enough to trigger bet (> 1/2.0 = 0.50)
        probas = np.array([0.60, 0.60])
        
        roi = optimizer._calculate_roi(df, probas)
        
        assert roi == pytest.approx(0.0)

    def test_calculate_roi_all_wins(self):
        """
        Test ROI calc:
        1. Bet Won @ 3.0 -> Profit +2
        Total Staked: 1
        Total Profit: 2
        ROI: 200%
        """
        optimizer = ProfitOptimizer(feature_cols=["f1"])
        
        df = pl.DataFrame({
            "f1": [1],
            "odds_player": [3.0],
            "player_won": [1]
        })
        
        probas = np.array([0.60]) # > 1/3.0 = 0.33
        
        roi = optimizer._calculate_roi(df, probas)
        
        assert roi == pytest.approx(200.0)

    def test_calculate_roi_no_bets(self):
        """
        Test ROI calc when probabilities are too low to bet.
        """
        optimizer = ProfitOptimizer(feature_cols=["f1"])
        
        df = pl.DataFrame({
            "f1": [1],
            "odds_player": [2.0],
            "player_won": [1]
        })
        
        probas = np.array([0.40]) # < 0.50, so no bet
        
        roi = optimizer._calculate_roi(df, probas)
        
        assert roi == 0.0

    def test_calculate_roi_mixed(self):
        """
        1. Win @ 2.0 (+1)
        2. Loss @ 2.0 (-1)
        3. Win @ 1.5 (+0.5)
        Total Profit: 0.5
        Total Stake: 3
        ROI: 0.5 / 3 = 16.66%
        """
        optimizer = ProfitOptimizer(feature_cols=["f1"])
        
        df = pl.DataFrame({
            "odds_player": [2.0, 2.0, 1.5],
            "player_won": [1, 0, 1]
        })
        
        # Ensure all bet
        probas = np.array([0.9, 0.9, 0.9])
        
        roi = optimizer._calculate_roi(df, probas)
        
        expected = (0.5 / 3.0) * 100
        assert roi == pytest.approx(expected)
