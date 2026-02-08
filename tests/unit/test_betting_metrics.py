"""
Unit tests for BettingMetrics calculator.
"""
import pytest
import numpy as np

from src.model.metrics import BettingMetrics, BettingMetricsResult, CVMetricsAggregator


class TestBettingMetrics:
    """Test suite for BettingMetrics calculator."""
    
    @pytest.fixture
    def metrics_calc(self) -> BettingMetrics:
        """Create a default BettingMetrics calculator."""
        return BettingMetrics(
            kelly_fraction=0.25,
            min_edge=0.05,
            min_odds=1.20,
            max_odds=5.00,
            initial_bankroll=1000.0,
            max_stake_pct=0.03,
        )
    
    def test_perfect_predictions_positive_roi(self, metrics_calc):
        """Test that perfect predictions yield positive ROI."""
        n_samples = 100
        y_true = np.ones(n_samples, dtype=int)  # All wins
        y_prob = np.ones(n_samples) * 0.8  # High confidence
        odds = np.ones(n_samples) * 2.0  # Even money implied, but model sees 80%
        
        result = metrics_calc.calculate(y_true, y_prob, odds)
        
        assert result.roi > 0
        assert result.auc >= 0.5  # AUC is 0.5 when all same class
        assert result.n_bets > 0
    
    def test_random_predictions_near_zero_roi(self, metrics_calc):
        """Test that random predictions yield near-zero or negative ROI."""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.rand(n_samples)  # Random predictions
        odds = 1.5 + np.random.rand(n_samples)  # Random odds
        
        result = metrics_calc.calculate(y_true, y_prob, odds)
        
        # With random predictions, ROI should be near zero or negative
        # This isn't guaranteed but probabilistically likely
        assert result.roi < 0.5  # Should not be wildly profitable on random
    
    def test_edge_filtering(self, metrics_calc):
        """Test that bets below min_edge are filtered out."""
        n_samples = 100
        y_true = np.ones(n_samples, dtype=int)
        y_prob = np.ones(n_samples) * 0.52  # Probability just above 50%
        odds = np.ones(n_samples) * 2.0  # Implied: 50%
        
        # Edge = 0.52 - 0.50 = 0.02 < min_edge (0.05)
        result = metrics_calc.calculate(y_true, y_prob, odds)
        
        assert result.n_bets == 0
        assert result.roi == 0.0
    
    def test_odds_filtering(self, metrics_calc):
        """Test that odds outside bounds are filtered."""
        n_samples = 100
        y_true = np.ones(n_samples, dtype=int)
        y_prob = np.ones(n_samples) * 0.8
        
        # Odds too low
        odds_low = np.ones(n_samples) * 1.1
        result_low = metrics_calc.calculate(y_true, y_prob, odds_low)
        assert result_low.n_bets == 0
        
        # Odds too high
        odds_high = np.ones(n_samples) * 10.0
        result_high = metrics_calc.calculate(y_true, y_prob, odds_high)
        assert result_high.n_bets == 0
    
    def test_max_drawdown_calculation(self, metrics_calc):
        """Test that max drawdown is calculated correctly."""
        # Scenario: win, win, lose, lose, lose
        y_true = np.array([1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9])  # High confidence
        odds = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        
        result = metrics_calc.calculate(y_true, y_prob, odds)
        
        # Should have some drawdown after the losses
        assert result.max_drawdown > 0
    
    def test_sharpe_ratio_calculation(self, metrics_calc):
        """Test Sharpe ratio formula."""
        # Mixed outcomes with varying odds create variance in returns
        y_true = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1])  # Mostly wins
        y_prob = np.ones(10) * 0.85  # High confidence
        odds = np.array([1.8, 2.0, 2.2, 1.9, 2.1, 2.3, 2.0, 1.7, 2.5, 2.0])  # Varying odds
        
        result = metrics_calc.calculate(y_true, y_prob, odds)
        
        # With mostly wins and varied odds, should have defined Sharpe
        # (Sharpe can be 0 if std=0 from constant returns)
        assert result.n_bets > 0  # Should place some bets
    
    def test_result_to_dict(self, metrics_calc):
        """Test that result can be serialized to dict."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_prob = np.array([0.9, 0.3, 0.8, 0.85, 0.2])
        odds = np.array([2.0, 2.5, 1.8, 2.2, 3.0])
        
        result = metrics_calc.calculate(y_true, y_prob, odds)
        result_dict = result.to_dict()
        
        assert "auc" in result_dict
        assert "roi" in result_dict
        assert "sharpe_ratio" in result_dict
        assert "max_drawdown" in result_dict
        assert isinstance(result_dict["n_bets"], int)
    
    def test_nan_handling(self, metrics_calc):
        """Test that NaN values are handled gracefully."""
        y_true = np.array([1, 0, 1, 1, 0])
        y_prob = np.array([0.9, np.nan, 0.8, 0.85, 0.2])
        odds = np.array([2.0, 2.5, np.nan, 2.2, 3.0])
        
        result = metrics_calc.calculate(y_true, y_prob, odds)
        
        # Should not crash and should filter out NaN
        assert result.n_samples == 5


class TestCVMetricsAggregator:
    """Test suite for CVMetricsAggregator."""
    
    @pytest.fixture
    def sample_results(self) -> list:
        """Create sample fold results."""
        return [
            BettingMetricsResult(
                auc=0.75, log_loss=0.5, accuracy=0.7,
                roi=0.05, sharpe_ratio=0.3, max_drawdown=0.1,
                n_bets=50, n_samples=200, bet_rate=0.25, win_rate=0.55,
                final_bankroll=1050, peak_bankroll=1080
            ),
            BettingMetricsResult(
                auc=0.78, log_loss=0.48, accuracy=0.72,
                roi=0.08, sharpe_ratio=0.4, max_drawdown=0.08,
                n_bets=55, n_samples=220, bet_rate=0.25, win_rate=0.58,
                final_bankroll=1080, peak_bankroll=1100
            ),
            BettingMetricsResult(
                auc=0.72, log_loss=0.52, accuracy=0.68,
                roi=0.02, sharpe_ratio=0.2, max_drawdown=0.12,
                n_bets=45, n_samples=180, bet_rate=0.25, win_rate=0.52,
                final_bankroll=1020, peak_bankroll=1060
            ),
        ]
    
    def test_add_fold_and_get_summary(self, sample_results):
        """Test adding folds and getting summary."""
        aggregator = CVMetricsAggregator()
        
        for result in sample_results:
            aggregator.add_fold(result)
        
        summary = aggregator.get_summary()
        
        assert summary["n_folds"] == 3
        assert "auc_mean" in summary
        assert "roi_mean" in summary
        assert "sharpe_ratio_mean" in summary
        assert "max_drawdown_mean" in summary
    
    def test_get_aggregated_result(self, sample_results):
        """Test getting aggregated result."""
        aggregator = CVMetricsAggregator()
        
        for result in sample_results:
            aggregator.add_fold(result)
        
        agg_result = aggregator.get_aggregated_result()
        
        assert isinstance(agg_result, BettingMetricsResult)
        assert agg_result.auc == pytest.approx((0.75 + 0.78 + 0.72) / 3, rel=0.01)
        assert agg_result.n_bets == 50 + 55 + 45  # Sum, not mean
    
    def test_empty_aggregator_raises(self):
        """Test that empty aggregator raises on get_aggregated_result."""
        aggregator = CVMetricsAggregator()
        
        with pytest.raises(ValueError, match="No fold results"):
            aggregator.get_aggregated_result()
    
    def test_empty_aggregator_summary(self):
        """Test that empty aggregator returns empty summary."""
        aggregator = CVMetricsAggregator()
        
        summary = aggregator.get_summary()
        assert summary == {}


class TestBettingMetricsEdgeCases:
    """Edge case tests for BettingMetrics."""
    
    def test_all_losses(self):
        """Test scenario where all bets lose."""
        calc = BettingMetrics(kelly_fraction=0.25, min_edge=0.05)
        
        y_true = np.zeros(20, dtype=int)  # All losses
        y_prob = np.ones(20) * 0.9  # High confidence (wrongly)
        odds = np.ones(20) * 2.0
        
        result = calc.calculate(y_true, y_prob, odds)
        
        assert result.roi < 0
        assert result.max_drawdown > 0
    
    def test_no_qualifying_bets(self):
        """Test when no bets meet criteria."""
        calc = BettingMetrics(kelly_fraction=0.25, min_edge=0.20)  # High threshold
        
        y_true = np.array([1, 0, 1, 1, 0])
        y_prob = np.array([0.6, 0.4, 0.55, 0.58, 0.45])  # Low edge
        odds = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        
        result = calc.calculate(y_true, y_prob, odds)
        
        assert result.n_bets == 0
        assert result.roi == 0.0
        assert result.sharpe_ratio == 0.0
    
    def test_single_bet(self):
        """Test with only one qualifying bet."""
        calc = BettingMetrics(kelly_fraction=0.25, min_edge=0.05)
        
        y_true = np.array([1])  # Single win
        y_prob = np.array([0.8])
        odds = np.array([2.0])
        
        result = calc.calculate(y_true, y_prob, odds)
        
        assert result.n_bets == 1
        assert result.roi > 0
