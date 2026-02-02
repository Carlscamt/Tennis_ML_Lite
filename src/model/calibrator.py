"""
Probability calibration for tennis match predictions.

Implements isotonic regression to fix model underconfidence, 
particularly in the 50-60% probability range.
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)


# Probability-based minimum EV thresholds
# Higher EV required for lower probability bets (more variance)
EV_THRESHOLDS: Dict[float, float] = {
    0.35: 0.040,  # 4.0% minimum EV for 35-40%
    0.40: 0.035,  # 3.5%
    0.50: 0.030,  # 3.0%
    0.60: 0.025,  # 2.5%
    0.70: 0.020,  # 2.0%
    0.80: 0.015,  # 1.5%
}

# Don't bet below this calibrated probability (sparse data)
MIN_PROB_THRESHOLD = 0.35


class ProbabilityCalibrator:
    """
    Isotonic calibration for probability predictions.
    
    Fixes systematic biases like underconfidence in the 50-60% range
    by learning a monotonic mapping from raw probabilities to 
    calibrated probabilities.
    """
    
    def __init__(self):
        self.isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        self.is_fitted = False
        self.calibration_stats = {}
    
    def fit(self, raw_probs: np.ndarray, actual_outcomes: np.ndarray) -> 'ProbabilityCalibrator':
        """
        Fit the isotonic calibrator on validation data.
        
        Args:
            raw_probs: Model's raw probability predictions
            actual_outcomes: Actual binary outcomes (0 or 1)
            
        Returns:
            Self for chaining
        """
        raw_probs = np.asarray(raw_probs).flatten()
        actual_outcomes = np.asarray(actual_outcomes).flatten()
        
        # Fit isotonic regression
        self.isotonic.fit(raw_probs, actual_outcomes)
        self.is_fitted = True
        
        # Calculate calibration stats for logging
        calibrated = self.calibrate(raw_probs)
        
        self.calibration_stats = {
            'n_samples': len(raw_probs),
            'raw_mean': float(raw_probs.mean()),
            'calibrated_mean': float(calibrated.mean()),
            'actual_mean': float(actual_outcomes.mean()),
            'max_adjustment': float(np.abs(calibrated - raw_probs).max()),
        }
        
        logger.info(
            f"Calibrator fitted on {len(raw_probs)} samples. "
            f"Max adjustment: {self.calibration_stats['max_adjustment']:.3f}"
        )
        
        return self
    
    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to raw probabilities.
        
        Args:
            raw_probs: Model's raw probability predictions
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw probabilities")
            return np.asarray(raw_probs)
        
        raw_probs = np.asarray(raw_probs).flatten()
        return self.isotonic.predict(raw_probs)
    
    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        path = Path(path)
        data = {
            'isotonic': self.isotonic,
            'is_fitted': self.is_fitted,
            'calibration_stats': self.calibration_stats,
        }
        joblib.dump(data, path)
        logger.info(f"Calibrator saved to {path}")
    
    def load(self, path: Path) -> 'ProbabilityCalibrator':
        """Load calibrator from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Calibrator not found at {path}")
            return self
            
        data = joblib.load(path)
        self.isotonic = data['isotonic']
        self.is_fitted = data['is_fitted']
        self.calibration_stats = data.get('calibration_stats', {})
        logger.info(f"Calibrator loaded from {path}")
        return self


def get_min_ev_threshold(probability: float) -> float:
    """
    Get the minimum EV threshold for a given probability level.
    
    Higher required EV for lower probabilities to account for variance.
    
    Args:
        probability: Calibrated probability
        
    Returns:
        Minimum EV required to place bet
    """
    # Find the appropriate threshold
    for prob_level, min_ev in sorted(EV_THRESHOLDS.items(), reverse=True):
        if probability >= prob_level:
            return min_ev
    
    # Below all thresholds - use highest EV requirement
    return max(EV_THRESHOLDS.values())


def passes_ev_gate(probability: float, edge: float) -> bool:
    """
    Check if a bet passes the probability-based EV gate.
    
    Args:
        probability: Calibrated probability
        edge: Calculated edge (model_prob - implied_prob)
        
    Returns:
        True if bet passes EV gate
    """
    # First check: minimum probability threshold
    if probability < MIN_PROB_THRESHOLD:
        return False
    
    # Second check: minimum EV by probability
    min_ev = get_min_ev_threshold(probability)
    return edge >= min_ev
