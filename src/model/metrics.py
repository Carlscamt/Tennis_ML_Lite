"""
Betting-specific metrics for model evaluation.

Computes ROI, Sharpe ratio, max drawdown, and other metrics
relevant to betting performance rather than just prediction accuracy.
"""
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class BettingMetricsResult:
    """Results from betting simulation on a single fold or test set."""
    # Classification metrics
    auc: float
    log_loss: float
    accuracy: float

    # Betting metrics
    roi: float                  # Return on Investment (profit / total_staked)
    sharpe_ratio: float         # Risk-adjusted return: mean(returns) / std(returns)
    max_drawdown: float         # Maximum peak-to-trough decline

    # Volume metrics
    n_bets: int                 # Number of bets placed
    n_samples: int              # Total samples evaluated
    bet_rate: float             # Fraction of samples that became bets
    win_rate: float             # Win rate of placed bets

    # Bankroll tracking
    final_bankroll: float       # Final bankroll value
    peak_bankroll: float        # Maximum bankroll reached

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "auc": round(self.auc, 4),
            "log_loss": round(self.log_loss, 4),
            "accuracy": round(self.accuracy, 4),
            "roi": round(self.roi, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "n_bets": self.n_bets,
            "n_samples": self.n_samples,
            "bet_rate": round(self.bet_rate, 4),
            "win_rate": round(self.win_rate, 4),
            "final_bankroll": round(self.final_bankroll, 2),
            "peak_bankroll": round(self.peak_bankroll, 2),
        }


@dataclass
class BettingMetrics:
    """
    Calculate betting-relevant metrics from predictions.
    
    Simulates betting using Kelly criterion and EV-based bet selection.
    
    Example:
        calc = BettingMetrics(kelly_fraction=0.25, min_edge=0.05)
        result = calc.calculate(y_true, y_prob, odds)
        print(f"ROI: {result.roi:.2%}, Sharpe: {result.sharpe_ratio:.2f}")
    """
    kelly_fraction: float = 0.25    # Fractional Kelly (1/4 for safety)
    min_edge: float = 0.05          # Minimum edge to place bet (5%)
    min_odds: float = 1.20          # Minimum acceptable odds
    max_odds: float = 5.00          # Maximum acceptable odds
    initial_bankroll: float = 1000.0
    max_stake_pct: float = 0.03     # Max 3% of bankroll per bet

    def calculate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        odds: np.ndarray,
        return_trajectory: bool = False
    ) -> BettingMetricsResult:
        """
        Simulate betting and compute comprehensive metrics.
        
        Args:
            y_true: Actual outcomes (0/1)
            y_prob: Model probability predictions
            odds: Decimal odds for the predicted outcome
            return_trajectory: If True, include bankroll trajectory in result
            
        Returns:
            BettingMetricsResult with all computed metrics
        """
        # Ensure numpy arrays
        y_true = np.asarray(y_true).flatten()
        y_prob = np.asarray(y_prob).flatten()
        odds = np.asarray(odds).flatten()

        n_samples = len(y_true)

        # Calculate classification metrics
        auc = self._safe_auc(y_true, y_prob)
        log_loss = self._safe_log_loss(y_true, y_prob)
        accuracy = np.mean((y_prob >= 0.5).astype(int) == y_true)

        # Filter to valid bets based on edge and odds constraints
        edges = y_prob - (1.0 / odds)
        bet_mask = (
            (edges >= self.min_edge) &
            (odds >= self.min_odds) &
            (odds <= self.max_odds) &
            (~np.isnan(odds)) &
            (~np.isnan(y_prob))
        )

        n_bets = int(np.sum(bet_mask))

        if n_bets == 0:
            # No bets placed - return zero betting metrics
            return BettingMetricsResult(
                auc=auc,
                log_loss=log_loss,
                accuracy=accuracy,
                roi=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                n_bets=0,
                n_samples=n_samples,
                bet_rate=0.0,
                win_rate=0.0,
                final_bankroll=self.initial_bankroll,
                peak_bankroll=self.initial_bankroll,
            )

        # Extract bet data
        bet_probs = y_prob[bet_mask]
        bet_odds = odds[bet_mask]
        bet_outcomes = y_true[bet_mask]

        # Calculate Kelly stakes for each bet
        stakes = self._kelly_stakes(bet_probs, bet_odds)

        # Simulate betting trajectory
        bankroll = self.initial_bankroll
        trajectory = [bankroll]
        bet_returns = []

        for prob, odd, outcome, stake_pct in zip(bet_probs, bet_odds, bet_outcomes, stakes):
            if stake_pct <= 0:
                continue

            stake = bankroll * stake_pct

            if outcome == 1:
                profit = stake * (odd - 1)
            else:
                profit = -stake

            bet_return = profit / stake if stake > 0 else 0
            bet_returns.append(bet_return)

            bankroll += profit
            trajectory.append(bankroll)

            # Bust protection
            if bankroll <= 0:
                bankroll = 0
                break

        trajectory = np.array(trajectory)
        bet_returns = np.array(bet_returns)

        # Calculate betting metrics
        total_staked = np.sum(stakes) * self.initial_bankroll
        total_profit = bankroll - self.initial_bankroll

        roi = total_profit / total_staked if total_staked > 0 else 0.0

        # Sharpe ratio (annualized if needed, but here per-bet)
        if len(bet_returns) > 1 and np.std(bet_returns) > 0:
            sharpe_ratio = np.mean(bet_returns) / np.std(bet_returns)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(trajectory)
        drawdown = (peak - trajectory) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Win rate
        wins = np.sum(bet_outcomes == 1)
        win_rate = wins / n_bets if n_bets > 0 else 0.0

        return BettingMetricsResult(
            auc=auc,
            log_loss=log_loss,
            accuracy=accuracy,
            roi=roi,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            n_bets=n_bets,
            n_samples=n_samples,
            bet_rate=n_bets / n_samples,
            win_rate=win_rate,
            final_bankroll=bankroll,
            peak_bankroll=float(np.max(trajectory)),
        )

    def _kelly_stakes(self, probs: np.ndarray, odds: np.ndarray) -> np.ndarray:
        """
        Calculate Kelly stake fractions for each bet.
        
        Kelly formula: f* = (b*p - q) / b
        where b = odds - 1, p = win prob, q = 1 - p
        """
        b = odds - 1  # Net odds
        p = probs
        q = 1 - p

        full_kelly = (b * p - q) / b
        fractional_kelly = full_kelly * self.kelly_fraction

        # Apply limits
        stakes = np.clip(fractional_kelly, 0, self.max_stake_pct)

        return stakes

    def _safe_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate AUC with error handling."""
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) < 2:
                return 0.5  # No discrimination possible
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            return 0.5

    def _safe_log_loss(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate log loss with error handling."""
        try:
            from sklearn.metrics import log_loss
            # Clip probabilities to avoid log(0)
            y_prob_clipped = np.clip(y_prob, 1e-15, 1 - 1e-15)
            return float(log_loss(y_true, y_prob_clipped))
        except Exception:
            return 1.0  # Worst case


@dataclass
class CVMetricsAggregator:
    """
    Aggregates BettingMetricsResult across multiple CV folds.
    """
    fold_results: list[BettingMetricsResult] = field(default_factory=list)

    def add_fold(self, result: BettingMetricsResult) -> None:
        """Add a fold's results to the aggregator."""
        self.fold_results.append(result)

    def get_summary(self) -> dict[str, Any]:
        """
        Compute summary statistics across all folds.
        
        Returns:
            Dictionary with mean, std, min, max for key metrics
        """
        if not self.fold_results:
            return {}

        metrics = {
            "auc": [r.auc for r in self.fold_results],
            "log_loss": [r.log_loss for r in self.fold_results],
            "roi": [r.roi for r in self.fold_results],
            "sharpe_ratio": [r.sharpe_ratio for r in self.fold_results],
            "max_drawdown": [r.max_drawdown for r in self.fold_results],
            "n_bets": [r.n_bets for r in self.fold_results],
            "win_rate": [r.win_rate for r in self.fold_results],
        }

        summary = {"n_folds": len(self.fold_results)}

        for name, values in metrics.items():
            arr = np.array(values)
            summary[f"{name}_mean"] = float(np.mean(arr))
            summary[f"{name}_std"] = float(np.std(arr))
            summary[f"{name}_min"] = float(np.min(arr))
            summary[f"{name}_max"] = float(np.max(arr))

        return summary

    def get_aggregated_result(self) -> BettingMetricsResult:
        """
        Return a single BettingMetricsResult with mean values.
        
        Useful for registration in model registry.
        """
        if not self.fold_results:
            raise ValueError("No fold results to aggregate")

        return BettingMetricsResult(
            auc=float(np.mean([r.auc for r in self.fold_results])),
            log_loss=float(np.mean([r.log_loss for r in self.fold_results])),
            accuracy=float(np.mean([r.accuracy for r in self.fold_results])),
            roi=float(np.mean([r.roi for r in self.fold_results])),
            sharpe_ratio=float(np.mean([r.sharpe_ratio for r in self.fold_results])),
            max_drawdown=float(np.mean([r.max_drawdown for r in self.fold_results])),
            n_bets=int(np.sum([r.n_bets for r in self.fold_results])),
            n_samples=int(np.sum([r.n_samples for r in self.fold_results])),
            bet_rate=float(np.mean([r.bet_rate for r in self.fold_results])),
            win_rate=float(np.mean([r.win_rate for r in self.fold_results])),
            final_bankroll=float(np.mean([r.final_bankroll for r in self.fold_results])),
            peak_bankroll=float(np.mean([r.peak_bankroll for r in self.fold_results])),
        )
