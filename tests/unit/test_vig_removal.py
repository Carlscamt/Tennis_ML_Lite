"""
Unit tests for Market Blending and Vig Removal logic.
"""
import pytest
import polars as pl
import numpy as np
from src.betting.signals import ValueBetFinder

def test_value_bet_finder_no_vig():
    """
    Test logic when there is NO vig (Fair Odds).
    Odds 2.0 vs 2.0 -> Implied 50% vs 50% -> Overround 1.0
    """
    finder = ValueBetFinder(blend_weight=0.5, min_edge=0.0, min_confidence=0.0)
    
    df = pl.DataFrame({
        "model_prob": [0.60],
        "odds_player": [2.0],
        "odds_opponent": [2.0]
    })
    
    result = finder.find_value_bets(df)
    
    # Assertions
    # Implied Prob = 0.5
    # Overround = 1.0
    # Fair Market = 0.5
    # Blended = 0.5 * 0.60 + 0.5 * 0.50 = 0.55
    # Edge = 0.55 - 0.50 = 0.05
    # EV = 0.55 * (2.0 - 1) - 0.45 = 0.10
    
    row = result.row(0, named=True)
    
    assert row["overround"] == pytest.approx(1.0)
    assert row["fair_market_prob"] == pytest.approx(0.5)
    assert row["blended_prob"] == pytest.approx(0.55)
    assert row["edge"] == pytest.approx(0.05)
    assert row["expected_value"] == pytest.approx(0.10)

def test_value_bet_finder_with_vig():
    """
    Test logic WITH vig.
    Odds 1.909 vs 1.909 -> Implied 0.5238 each -> Overround ~1.0476
    Fair prob should normalize back to 0.50
    """
    finder = ValueBetFinder(blend_weight=0.5, min_edge=0.0, min_confidence=0.0)
    
    odds = 1.90909  # 1 / 1.90909 = 0.5238
    
    df = pl.DataFrame({
        "model_prob": [0.60],
        "odds_player": [odds],
        "odds_opponent": [odds]
    })
    
    result = finder.find_value_bets(df)
    
    row = result.row(0, named=True)
    
    # Implied = 0.5238
    # Overround = 1.0476
    # Fair Market = 0.5238 / 1.0476 = 0.50
    
    assert row["implied_prob"] == pytest.approx(0.5238, abs=0.0001)
    assert row["overround"] == pytest.approx(1.0476, abs=0.0001)
    assert row["fair_market_prob"] == pytest.approx(0.50, abs=0.0001)
    
    # Blended = 0.5 * 0.60 + 0.5 * 0.50 = 0.55
    assert row["blended_prob"] == pytest.approx(0.55, abs=0.0001)
    
    # Edge = Blended - Implied (Cost basis is the bad price)
    # Edge = 0.55 - 0.5238 = 0.0262
    expected_edge = 0.55 - (1/odds)
    assert row["edge"] == pytest.approx(expected_edge, abs=0.0001)

def test_value_bet_finder_filtering():
    """Test min_edge and min_confidence filtering"""
    finder = ValueBetFinder(
        blend_weight=1.0,  # 100% Model for simplicity
        min_edge=0.10,     # Need 10% edge
        min_confidence=0.60
    )
    
    df = pl.DataFrame({
        "model_prob": [0.55, 0.70],     # 1: Low Conf, 2: High Conf
        "odds_player": [2.0, 2.0],      # 1: Edge 0.05, 2: Edge 0.20
        "odds_opponent": [2.0, 2.0]
    })
    
    result = finder.find_value_bets(df)
    
    assert len(result) == 1
    assert result["model_prob"][0] == 0.70
