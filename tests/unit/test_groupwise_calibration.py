"""
Unit tests for GroupWiseCalibrator and calibration methods.
"""
import pytest
import numpy as np

from src.model.calibrator import (
    ProbabilityCalibrator,
    CalibrationMethod,
    PlattCalibrator,
    GroupWiseCalibrator,
)


class TestPlattCalibrator:
    """Tests for PlattCalibrator (sigmoid scaling)."""
    
    def test_fit_and_calibrate(self):
        """Test basic Platt scaling."""
        platt = PlattCalibrator()
        
        # Simulated underconfident model
        raw_probs = np.random.uniform(0.4, 0.6, 100)
        # True outcomes correlate with raw probs
        outcomes = (raw_probs > 0.5).astype(int)
        
        platt.fit(raw_probs, outcomes)
        
        assert platt.is_fitted
        calibrated = platt.calibrate(np.array([0.55]))
        assert 0 <= calibrated[0] <= 1
    
    def test_not_fitted_returns_raw(self):
        """Test that unfitted calibrator returns raw probs."""
        platt = PlattCalibrator()
        raw = np.array([0.6, 0.7])
        
        result = platt.calibrate(raw)
        np.testing.assert_array_equal(result, raw)


class TestCalibrationMethod:
    """Tests for CalibrationMethod enum."""
    
    def test_enum_values(self):
        assert CalibrationMethod.ISOTONIC.value == "isotonic"
        assert CalibrationMethod.PLATT.value == "platt"
        assert CalibrationMethod.ENSEMBLE.value == "ensemble"


class TestGroupWiseCalibrator:
    """Tests for GroupWiseCalibrator."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data with groups."""
        np.random.seed(42)
        n = 2000
        
        raw_probs = np.random.uniform(0.4, 0.8, n)
        outcomes = (raw_probs + np.random.normal(0, 0.1, n) > 0.5).astype(int)
        genders = np.random.choice(["ATP", "WTA"], n)
        surfaces = np.random.choice(["hard", "clay", "grass"], n)
        odds = np.random.uniform(1.3, 4.0, n)
        
        return {
            "raw_probs": raw_probs,
            "outcomes": outcomes,
            "genders": genders,
            "surfaces": surfaces,
            "odds": odds,
        }
    
    def test_fit_creates_groups(self, sample_data):
        """Test that fitting creates calibration groups."""
        cal = GroupWiseCalibrator(min_group_samples=100)
        
        cal.fit(
            sample_data["raw_probs"],
            sample_data["outcomes"],
            genders=sample_data["genders"],
            surfaces=sample_data["surfaces"],
            odds=sample_data["odds"],
        )
        
        assert cal.is_fitted
        assert len(cal.groups) > 0
        assert cal.global_calibrator is not None
    
    def test_calibrate_returns_valid_probs(self, sample_data):
        """Test that calibrated probabilities are valid."""
        cal = GroupWiseCalibrator(min_group_samples=100)
        
        cal.fit(
            sample_data["raw_probs"],
            sample_data["outcomes"],
            genders=sample_data["genders"],
            surfaces=sample_data["surfaces"],
        )
        
        calibrated = cal.calibrate(
            sample_data["raw_probs"][:10],
            genders=sample_data["genders"][:10],
            surfaces=sample_data["surfaces"][:10],
        )
        
        assert len(calibrated) == 10
        assert all(0 <= p <= 1 for p in calibrated)
    
    def test_hierarchical_fallback(self, sample_data):
        """Test that hierarchical fallback works."""
        # Set very high min_samples so most groups fall back
        cal = GroupWiseCalibrator(min_group_samples=5000)
        
        cal.fit(
            sample_data["raw_probs"],
            sample_data["outcomes"],
            genders=sample_data["genders"],
            surfaces=sample_data["surfaces"],
        )
        
        # Few groups should be fitted, but global should exist
        assert cal.global_calibrator is not None
        assert cal.global_calibrator.is_fitted
        
        # Should still produce valid output via fallback
        calibrated = cal.calibrate(sample_data["raw_probs"][:5])
        assert len(calibrated) == 5
    
    def test_platt_method(self, sample_data):
        """Test with Platt scaling method."""
        cal = GroupWiseCalibrator(
            method=CalibrationMethod.PLATT,
            min_group_samples=100,
        )
        
        cal.fit(
            sample_data["raw_probs"],
            sample_data["outcomes"],
            genders=sample_data["genders"],
        )
        
        assert cal.is_fitted
        calibrated = cal.calibrate(sample_data["raw_probs"][:10])
        assert all(0 <= p <= 1 for p in calibrated)
    
    def test_ensemble_method(self, sample_data):
        """Test with ensemble method."""
        cal = GroupWiseCalibrator(
            method=CalibrationMethod.ENSEMBLE,
            min_group_samples=100,
        )
        
        cal.fit(
            sample_data["raw_probs"],
            sample_data["outcomes"],
            genders=sample_data["genders"],
        )
        
        assert cal.is_fitted
        calibrated = cal.calibrate(sample_data["raw_probs"][:10])
        assert all(0 <= p <= 1 for p in calibrated)
    
    def test_calibration_report(self, sample_data):
        """Test calibration report generation."""
        cal = GroupWiseCalibrator(min_group_samples=100)
        
        cal.fit(
            sample_data["raw_probs"],
            sample_data["outcomes"],
            genders=sample_data["genders"],
        )
        
        report = cal.get_calibration_report()
        
        assert "n_groups" in report
        assert "method" in report
        assert "groups" in report
        assert report["n_groups"] == len(report["groups"])
    
    def test_not_fitted_warning(self, sample_data):
        """Test that unfitted calibrator logs warning."""
        cal = GroupWiseCalibrator()
        
        calibrated = cal.calibrate(sample_data["raw_probs"][:5])
        
        # Should return raw probs unchanged
        np.testing.assert_array_almost_equal(
            calibrated, sample_data["raw_probs"][:5]
        )
    
    def test_odds_bucket_assignment(self):
        """Test odds bucket assignment."""
        cal = GroupWiseCalibrator()
        
        assert cal._get_odds_bucket(1.4) == "1.2-1.6"
        assert cal._get_odds_bucket(1.8) == "1.6-2.2"
        assert cal._get_odds_bucket(3.0) == "2.2-3.5"
        assert cal._get_odds_bucket(4.5) == "3.5-5.0"
        
        # Edge case: below range
        assert cal._get_odds_bucket(1.1) == "1.2-1.6"
        # Edge case: above range
        assert cal._get_odds_bucket(6.0) == "3.5-5.0"
