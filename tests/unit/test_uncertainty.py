"""
Unit tests for uncertainty signals and filtering.
"""
import pytest
import numpy as np

from src.model.uncertainty import (
    calculate_margin,
    calculate_entropy,
    calculate_confidence,
    calculate_uncertainty_signals,
    calculate_uncertainty_signals_batch,
    UncertaintyFilter,
    UncertaintySignals,
)


class TestUncertaintyCalculations:
    """Tests for uncertainty signal calculations."""
    
    def test_margin_at_boundary(self):
        """Test margin at decision boundary."""
        assert calculate_margin(0.5) == 0.0
    
    def test_margin_confident(self):
        """Test margin for confident predictions."""
        assert calculate_margin(0.8) == pytest.approx(0.3)
        assert calculate_margin(0.2) == pytest.approx(0.3)
    
    def test_margin_extreme(self):
        """Test margin at extremes."""
        assert calculate_margin(1.0) == pytest.approx(0.5)
        assert calculate_margin(0.0) == pytest.approx(0.5)
    
    def test_entropy_at_boundary(self):
        """Test entropy at maximum uncertainty."""
        # At p=0.5, entropy = 1.0 (max)
        assert calculate_entropy(0.5) == pytest.approx(1.0, rel=0.01)
    
    def test_entropy_confident(self):
        """Test entropy for confident predictions."""
        # More confident = lower entropy
        assert calculate_entropy(0.9) < calculate_entropy(0.7)
        assert calculate_entropy(0.1) < calculate_entropy(0.3)
    
    def test_entropy_extremes(self):
        """Test entropy near 0 and 1."""
        # Near 0 or 1 should have very low entropy
        assert calculate_entropy(0.99) < 0.1
        assert calculate_entropy(0.01) < 0.1
    
    def test_confidence_basic(self):
        """Test confidence calculation."""
        assert calculate_confidence(0.8) == 0.8
        assert calculate_confidence(0.2) == 0.8
        assert calculate_confidence(0.5) == 0.5
    
    def test_uncertainty_signals_dataclass(self):
        """Test full signal calculation."""
        signals = calculate_uncertainty_signals(0.7)
        
        assert isinstance(signals, UncertaintySignals)
        assert signals.margin == pytest.approx(0.2)
        assert signals.confidence == pytest.approx(0.7)
        assert signals.entropy > 0
    
    def test_signals_batch(self):
        """Test batch calculation."""
        probs = np.array([0.5, 0.7, 0.9])
        signals = calculate_uncertainty_signals_batch(probs)
        
        assert len(signals["margin"]) == 3
        assert len(signals["entropy"]) == 3
        assert len(signals["confidence"]) == 3
        
        assert signals["margin"][0] == pytest.approx(0.0)
        assert signals["margin"][1] == pytest.approx(0.2)
        assert signals["margin"][2] == pytest.approx(0.4)


class TestUncertaintyFilter:
    """Tests for UncertaintyFilter class."""
    
    @pytest.fixture
    def default_filter(self):
        return UncertaintyFilter()
    
    def test_passes_confident_prediction(self, default_filter):
        """Test that confident predictions pass."""
        # Need p >= 0.85 to have entropy < 0.65
        assert default_filter.passes_filter(0.90) is True
        assert default_filter.passes_filter(0.10) is True
    
    def test_rejects_boundary_prediction(self, default_filter):
        """Test that predictions near 0.5 are rejected."""
        assert default_filter.passes_filter(0.52) is False
        assert default_filter.passes_filter(0.48) is False
    
    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        strict_filter = UncertaintyFilter(
            min_margin=0.20,
            max_entropy=0.50,
            min_confidence=0.70,
        )
        
        # 0.65 has margin=0.15 < 0.20, should fail
        assert strict_filter.passes_filter(0.65) is False
        
        # 0.90 has margin=0.40, entropy~0.47, confidence=0.90, should pass
        assert strict_filter.passes_filter(0.90) is True
    
    def test_passes_filter_with_reason(self, default_filter):
        """Test rejection reason reporting."""
        passes, reason = default_filter.passes_filter_with_reason(0.52)
        
        assert passes is False
        assert "margin_too_low" in reason
    
    def test_filter_batch(self, default_filter):
        """Test batch filtering."""
        # Note: entropy at p=0.65 is ~0.93, which exceeds max_entropy=0.65
        # Need more confident predictions to pass the entropy filter
        probs = np.array([0.5, 0.55, 0.75, 0.85, 0.95])
        mask = default_filter.filter_batch(probs)
        
        # 0.5: margin=0, rejected (margin)
        # 0.55: margin=0.05, rejected (margin)
        # 0.75: margin=0.25, entropy~0.81, rejected (entropy)
        # 0.85: margin=0.35, entropy~0.61, passed
        # 0.95: margin=0.45, entropy~0.29, passed
        assert mask.tolist() == [False, False, False, True, True]
    
    def test_rejection_stats(self, default_filter):
        """Test rejection statistics."""
        probs = np.array([0.5, 0.55, 0.60, 0.65, 0.80])
        stats = default_filter.get_rejection_stats(probs)
        
        assert stats["total_samples"] == 5
        assert stats["total_rejections"] >= 2  # At least 0.5 and 0.55
        assert 0 <= stats["rejection_rate"] <= 1
