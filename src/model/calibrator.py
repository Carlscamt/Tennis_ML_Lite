"""
Probability calibration for tennis match predictions.

Implements isotonic regression to fix model underconfidence, 
particularly in the 50-60% probability range.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression


logger = logging.getLogger(__name__)


# Probability-based minimum EV thresholds
# Higher EV required for lower probability bets (more variance)
EV_THRESHOLDS: dict[float, float] = {
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


# =============================================================================
# CALIBRATION METHODS
# =============================================================================

from enum import Enum

from sklearn.linear_model import LogisticRegression


class CalibrationMethod(Enum):
    """Available calibration methods."""
    ISOTONIC = "isotonic"
    PLATT = "platt"  # Sigmoid/Platt scaling
    ENSEMBLE = "ensemble"  # Average of both


class PlattCalibrator:
    """
    Platt scaling (sigmoid calibration) for probability predictions.
    
    Uses logistic regression on raw probabilities to learn A and B 
    in: calibrated = 1 / (1 + exp(A * raw + B))
    """

    def __init__(self):
        self.lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, actual_outcomes: np.ndarray) -> 'PlattCalibrator':
        """Fit Platt scaling on validation data."""
        raw_probs = np.asarray(raw_probs).reshape(-1, 1)
        actual_outcomes = np.asarray(actual_outcomes).flatten()

        self.lr.fit(raw_probs, actual_outcomes)
        self.is_fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if not self.is_fitted:
            return np.asarray(raw_probs)

        raw_probs = np.asarray(raw_probs).reshape(-1, 1)
        return self.lr.predict_proba(raw_probs)[:, 1]


# =============================================================================
# GROUP-WISE CALIBRATION
# =============================================================================


@dataclass
class CalibrationGroup:
    """Represents a calibration group with its data and calibrator."""
    name: str
    calibrator: Optional['ProbabilityCalibrator'] = None
    platt_calibrator: PlattCalibrator | None = None
    sample_count: int = 0
    ece: float = 0.0
    brier_score: float = 0.0



@dataclass
class GroupWiseCalibrator:
    """
    Calibrator that fits separate models for different groups.
    
    Groups are defined by combinations of:
    - Gender (ATP/WTA)
    - Surface (hard, clay, grass, indoor_hard)
    - Odds range (1.2-1.6, 1.6-2.2, 2.2-3.5, 3.5-5.0)
    
    Uses hierarchical fallback when groups are too sparse:
    1. Full group (gender × surface × odds_range)
    2. Partial (gender × surface)
    3. Gender only
    4. Global calibrator
    """
    method: CalibrationMethod = CalibrationMethod.ISOTONIC
    min_group_samples: int = 500
    fallback_to_global: bool = True

    # Odds range buckets
    odds_buckets: tuple[tuple[float, float], ...] = (
        (1.2, 1.6),
        (1.6, 2.2),
        (2.2, 3.5),
        (3.5, 5.0),
    )

    # Fitted state
    groups: dict[str, CalibrationGroup] = field(default_factory=dict)
    global_calibrator: ProbabilityCalibrator | None = None
    global_platt: PlattCalibrator | None = None
    is_fitted: bool = False

    def _get_odds_bucket(self, odds: float) -> str:
        """Determine which odds bucket a value falls into."""
        for low, high in self.odds_buckets:
            if low <= odds < high:
                return f"{low}-{high}"
        # Default for out-of-range
        if odds < self.odds_buckets[0][0]:
            return f"{self.odds_buckets[0][0]}-{self.odds_buckets[0][1]}"
        return f"{self.odds_buckets[-1][0]}-{self.odds_buckets[-1][1]}"

    def _build_group_key(
        self,
        gender: str | None,
        surface: str | None,
        odds: float | None
    ) -> str:
        """Build a group key from components."""
        parts = []
        if gender:
            parts.append(gender.upper())
        if surface:
            parts.append(surface.lower())
        if odds is not None:
            parts.append(f"odds_{self._get_odds_bucket(odds)}")
        return "_".join(parts) if parts else "global"

    def _create_calibrator(self) -> tuple[ProbabilityCalibrator | None, PlattCalibrator | None]:
        """Create calibrator(s) based on method."""
        isotonic = None
        platt = None

        if self.method in (CalibrationMethod.ISOTONIC, CalibrationMethod.ENSEMBLE):
            isotonic = ProbabilityCalibrator()
        if self.method in (CalibrationMethod.PLATT, CalibrationMethod.ENSEMBLE):
            platt = PlattCalibrator()

        return isotonic, platt

    def fit(
        self,
        raw_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        genders: np.ndarray | None = None,
        surfaces: np.ndarray | None = None,
        odds: np.ndarray | None = None,
    ) -> 'GroupWiseCalibrator':
        """
        Fit group-wise calibrators.
        
        Args:
            raw_probs: Model probability predictions
            actual_outcomes: Actual binary outcomes
            genders: Optional array of 'ATP' or 'WTA'
            surfaces: Optional array of surface types
            odds: Optional array of decimal odds
        """
        raw_probs = np.asarray(raw_probs).flatten()
        actual_outcomes = np.asarray(actual_outcomes).flatten()
        n_samples = len(raw_probs)

        # Prepare group arrays
        genders = np.asarray(genders).flatten() if genders is not None else np.array(['UNKNOWN'] * n_samples)
        surfaces = np.asarray(surfaces).flatten() if surfaces is not None else np.array(['unknown'] * n_samples)
        odds = np.asarray(odds).flatten() if odds is not None else np.array([2.0] * n_samples)

        # 1. Fit global calibrator first (always available as fallback)
        self.global_calibrator, self.global_platt = self._create_calibrator()
        if self.global_calibrator:
            self.global_calibrator.fit(raw_probs, actual_outcomes)
        if self.global_platt:
            self.global_platt.fit(raw_probs, actual_outcomes)

        # 2. Build groups and fit calibrators for sufficiently large ones
        from collections import defaultdict
        group_data = defaultdict(lambda: {'probs': [], 'outcomes': [], 'indices': []})

        for i in range(n_samples):
            # Full key
            full_key = self._build_group_key(genders[i], surfaces[i], odds[i])
            group_data[full_key]['probs'].append(raw_probs[i])
            group_data[full_key]['outcomes'].append(actual_outcomes[i])
            group_data[full_key]['indices'].append(i)

            # Partial key (gender × surface)
            partial_key = self._build_group_key(genders[i], surfaces[i], None)
            group_data[partial_key]['probs'].append(raw_probs[i])
            group_data[partial_key]['outcomes'].append(actual_outcomes[i])

            # Gender-only key
            gender_key = self._build_group_key(genders[i], None, None)
            group_data[gender_key]['probs'].append(raw_probs[i])
            group_data[gender_key]['outcomes'].append(actual_outcomes[i])

        # 3. Fit calibrators for groups with sufficient data
        for key, data in group_data.items():
            n = len(data['probs'])
            if n >= self.min_group_samples:
                probs_arr = np.array(data['probs'])
                outcomes_arr = np.array(data['outcomes'])

                iso_cal, platt_cal = self._create_calibrator()
                if iso_cal:
                    iso_cal.fit(probs_arr, outcomes_arr)
                if platt_cal:
                    platt_cal.fit(probs_arr, outcomes_arr)

                # Calculate calibration metrics
                ece = self._calculate_ece(probs_arr, outcomes_arr, iso_cal, platt_cal)
                brier = self._calculate_brier(probs_arr, outcomes_arr, iso_cal, platt_cal)

                self.groups[key] = CalibrationGroup(
                    name=key,
                    calibrator=iso_cal,
                    platt_calibrator=platt_cal,
                    sample_count=n,
                    ece=ece,
                    brier_score=brier,
                )
                logger.info(f"Fitted calibrator for group '{key}' ({n} samples, ECE={ece:.4f})")
            else:
                logger.debug(f"Group '{key}' too small ({n} < {self.min_group_samples}), will use fallback")

        self.is_fitted = True
        logger.info(f"GroupWiseCalibrator fitted with {len(self.groups)} groups from {n_samples} samples")

        return self

    def calibrate(
        self,
        raw_probs: np.ndarray,
        genders: np.ndarray | None = None,
        surfaces: np.ndarray | None = None,
        odds: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply calibration with hierarchical fallback.
        
        Returns calibrated probabilities.
        """
        if not self.is_fitted:
            logger.warning("GroupWiseCalibrator not fitted, returning raw probabilities")
            return np.asarray(raw_probs)

        raw_probs = np.asarray(raw_probs).flatten()
        n_samples = len(raw_probs)

        genders = np.asarray(genders).flatten() if genders is not None else np.array(['UNKNOWN'] * n_samples)
        surfaces = np.asarray(surfaces).flatten() if surfaces is not None else np.array(['unknown'] * n_samples)
        odds = np.asarray(odds).flatten() if odds is not None else np.array([2.0] * n_samples)

        calibrated = np.zeros(n_samples)

        for i in range(n_samples):
            p = raw_probs[i]

            # Try hierarchical fallback
            keys_to_try = [
                self._build_group_key(genders[i], surfaces[i], odds[i]),  # Full
                self._build_group_key(genders[i], surfaces[i], None),     # Gender × Surface
                self._build_group_key(genders[i], None, None),            # Gender only
            ]

            calibrated_p = None
            for key in keys_to_try:
                if key in self.groups:
                    group = self.groups[key]
                    calibrated_p = self._apply_calibration(p, group.calibrator, group.platt_calibrator)
                    break

            # Fallback to global if no group found
            if calibrated_p is None and self.fallback_to_global:
                calibrated_p = self._apply_calibration(p, self.global_calibrator, self.global_platt)

            calibrated[i] = calibrated_p if calibrated_p is not None else p

        return calibrated

    def _apply_calibration(
        self,
        prob: float,
        iso_cal: ProbabilityCalibrator | None,
        platt_cal: PlattCalibrator | None
    ) -> float:
        """Apply calibration using configured method."""
        probs = []

        if iso_cal and iso_cal.is_fitted:
            probs.append(iso_cal.calibrate(np.array([prob]))[0])
        if platt_cal and platt_cal.is_fitted:
            probs.append(platt_cal.calibrate(np.array([prob]))[0])

        if not probs:
            return prob

        if self.method == CalibrationMethod.ENSEMBLE:
            return float(np.mean(probs))
        return probs[0]

    def _calculate_ece(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        iso_cal: ProbabilityCalibrator | None,
        platt_cal: PlattCalibrator | None,
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        calibrated = self._apply_calibration_array(probs, iso_cal, platt_cal)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (calibrated >= bin_boundaries[i]) & (calibrated < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(outcomes[mask])
                bin_confidence = np.mean(calibrated[mask])
                ece += np.sum(mask) * np.abs(bin_accuracy - bin_confidence)

        return ece / len(probs) if len(probs) > 0 else 0.0

    def _calculate_brier(
        self,
        probs: np.ndarray,
        outcomes: np.ndarray,
        iso_cal: ProbabilityCalibrator | None,
        platt_cal: PlattCalibrator | None,
    ) -> float:
        """Calculate Brier score."""
        calibrated = self._apply_calibration_array(probs, iso_cal, platt_cal)
        return float(np.mean((calibrated - outcomes) ** 2))

    def _apply_calibration_array(
        self,
        probs: np.ndarray,
        iso_cal: ProbabilityCalibrator | None,
        platt_cal: PlattCalibrator | None,
    ) -> np.ndarray:
        """Apply calibration to an array."""
        if iso_cal and iso_cal.is_fitted:
            return iso_cal.calibrate(probs)
        if platt_cal and platt_cal.is_fitted:
            return platt_cal.calibrate(probs)
        return probs

    def get_calibration_report(self) -> dict[str, any]:
        """Return summary statistics about calibration groups."""
        return {
            'n_groups': len(self.groups),
            'method': self.method.value,
            'groups': {
                name: {
                    'samples': g.sample_count,
                    'ece': round(g.ece, 4),
                    'brier': round(g.brier_score, 4),
                }
                for name, g in self.groups.items()
            }
        }

