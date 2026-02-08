"""
Uncertainty quantification for betting decisions.

Provides signals to filter out noisy predictions where the model
has high reported edge but low actual confidence.
"""
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class UncertaintySignals:
    """Container for uncertainty metrics on a single prediction."""
    margin: float           # |p - 0.5| — distance from decision boundary
    entropy: float          # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    confidence: float       # max(p, 1-p)

    def to_dict(self) -> dict[str, float]:
        return {
            "margin": round(self.margin, 4),
            "entropy": round(self.entropy, 4),
            "confidence": round(self.confidence, 4),
        }


def calculate_margin(prob: float) -> float:
    """
    Calculate margin from decision boundary.
    
    Margin = |p - 0.5|
    
    Higher margin = more confident prediction.
    Low margin (near 0.5) = uncertain, should be avoided for betting.
    
    Args:
        prob: Probability prediction
        
    Returns:
        Margin value in [0, 0.5]
    """
    return abs(prob - 0.5)


def calculate_entropy(prob: float, eps: float = 1e-10) -> float:
    """
    Calculate binary entropy of a probability.
    
    H(p) = -p*log2(p) - (1-p)*log2(1-p)
    
    Entropy ranges from 0 (certain) to 1 (maximum uncertainty at p=0.5).
    
    Args:
        prob: Probability prediction
        eps: Small value to avoid log(0)
        
    Returns:
        Entropy value in [0, 1]
    """
    p = np.clip(prob, eps, 1 - eps)
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def calculate_confidence(prob: float) -> float:
    """
    Simple confidence score: max(p, 1-p).
    
    Ranges from 0.5 (uncertain) to 1.0 (certain).
    """
    return max(prob, 1 - prob)


def calculate_uncertainty_signals(prob: float) -> UncertaintySignals:
    """
    Calculate all uncertainty signals for a probability.
    
    Args:
        prob: Model probability prediction
        
    Returns:
        UncertaintySignals dataclass
    """
    return UncertaintySignals(
        margin=calculate_margin(prob),
        entropy=calculate_entropy(prob),
        confidence=calculate_confidence(prob),
    )


def calculate_uncertainty_signals_batch(probs: np.ndarray) -> dict[str, np.ndarray]:
    """
    Calculate uncertainty signals for a batch of probabilities.
    
    Args:
        probs: Array of probability predictions
        
    Returns:
        Dictionary with 'margin', 'entropy', 'confidence' arrays
    """
    probs = np.asarray(probs).flatten()
    eps = 1e-10

    margin = np.abs(probs - 0.5)

    p_clipped = np.clip(probs, eps, 1 - eps)
    entropy = -p_clipped * np.log2(p_clipped) - (1 - p_clipped) * np.log2(1 - p_clipped)

    confidence = np.maximum(probs, 1 - probs)

    return {
        "margin": margin,
        "entropy": entropy,
        "confidence": confidence,
    }


@dataclass
class UncertaintyFilter:
    """
    Filter that rejects bets with high uncertainty.
    
    Even if a bet has high edge, if the probability is near 0.5
    (low margin) or has high entropy, it's more likely to be noise.
    
    Example:
        filter = UncertaintyFilter(min_margin=0.10, max_entropy=0.65)
        if filter.passes_filter(prob):
            # Place bet
    """
    min_margin: float = 0.10       # Reject if |p - 0.5| < 0.10 (p in [0.4, 0.6])
    max_entropy: float = 0.65      # Reject if entropy > 0.65
    min_confidence: float = 0.55   # Reject if max(p, 1-p) < 0.55

    def passes_filter(self, prob: float) -> bool:
        """
        Check if a probability passes the uncertainty filter.
        
        Args:
            prob: Calibrated probability
            
        Returns:
            True if prediction is confident enough to bet on
        """
        margin = calculate_margin(prob)
        entropy = calculate_entropy(prob)
        confidence = calculate_confidence(prob)

        if margin < self.min_margin:
            return False
        if entropy > self.max_entropy:
            return False
        if confidence < self.min_confidence:
            return False

        return True

    def passes_filter_with_reason(self, prob: float) -> tuple[bool, str | None]:
        """
        Check filter and return rejection reason if any.
        
        Returns:
            Tuple of (passes, rejection_reason)
        """
        margin = calculate_margin(prob)
        entropy = calculate_entropy(prob)
        confidence = calculate_confidence(prob)

        if margin < self.min_margin:
            return False, f"margin_too_low ({margin:.3f} < {self.min_margin})"
        if entropy > self.max_entropy:
            return False, f"entropy_too_high ({entropy:.3f} > {self.max_entropy})"
        if confidence < self.min_confidence:
            return False, f"confidence_too_low ({confidence:.3f} < {self.min_confidence})"

        return True, None

    def filter_batch(self, probs: np.ndarray) -> np.ndarray:
        """
        Return boolean mask for probabilities that pass the filter.
        
        Args:
            probs: Array of probabilities
            
        Returns:
            Boolean array where True = passes filter
        """
        signals = calculate_uncertainty_signals_batch(probs)

        mask = (
            (signals["margin"] >= self.min_margin) &
            (signals["entropy"] <= self.max_entropy) &
            (signals["confidence"] >= self.min_confidence)
        )

        return mask

    def get_rejection_stats(self, probs: np.ndarray) -> dict[str, Any]:
        """
        Get statistics on how many predictions would be rejected.
        
        Args:
            probs: Array of probabilities
            
        Returns:
            Dictionary with rejection statistics
        """
        probs = np.asarray(probs).flatten()
        signals = calculate_uncertainty_signals_batch(probs)

        margin_rejects = np.sum(signals["margin"] < self.min_margin)
        entropy_rejects = np.sum(signals["entropy"] > self.max_entropy)
        confidence_rejects = np.sum(signals["confidence"] < self.min_confidence)

        # Total rejections (any condition)
        total_rejects = np.sum(~self.filter_batch(probs))

        return {
            "total_samples": len(probs),
            "total_rejections": int(total_rejects),
            "rejection_rate": float(total_rejects / len(probs)) if len(probs) > 0 else 0,
            "margin_rejections": int(margin_rejects),
            "entropy_rejections": int(entropy_rejects),
            "confidence_rejections": int(confidence_rejects),
        }
