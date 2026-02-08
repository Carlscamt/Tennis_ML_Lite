"""
Risk-of-ruin diagnostics and Monte Carlo simulation.
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DrawdownStats:
    """Statistics from Monte Carlo drawdown simulation."""
    mean_max_drawdown: float
    median_max_drawdown: float
    percentile_95: float
    percentile_99: float
    worst_drawdown: float
    ruin_count: int
    ruin_probability: float
    simulations: int


@dataclass
class BetRecord:
    """Historical bet record for simulation."""
    edge: float
    odds: float
    stake_pct: float
    won: bool


class RiskAnalyzer:
    """
    Risk-of-ruin and drawdown analysis.
    
    Provides:
    - Analytical ruin probability estimates
    - Monte Carlo drawdown simulations
    - Kelly fraction risk analysis
    """
    
    def ruin_probability(
        self,
        edge: float,
        win_prob: float,
        kelly_fraction: float,
        n_bets: int,
        ruin_threshold: float = 0.1
    ) -> float:
        """
        Estimate probability of ruin using analytical approximation.
        
        Uses the formula for fractional Kelly:
        P(ruin) ≈ (1 - edge)^(bankroll/stake) for simplified case
        
        More accurate for fractional Kelly with many bets.
        
        Args:
            edge: Expected edge (e.g., 0.05 for 5%)
            win_prob: Win probability
            kelly_fraction: Kelly fraction used (e.g., 0.25)
            n_bets: Number of bets
            ruin_threshold: Bankroll fraction considered ruin (default 10%)
            
        Returns:
            Estimated ruin probability
        """
        if edge <= 0:
            return 1.0  # Negative edge = certain ruin eventually
        
        if kelly_fraction <= 0:
            return 0.0  # No betting = no ruin
        
        # Average odds implied by edge and win prob
        # edge = p * odds - 1, so odds = (edge + 1) / p
        avg_odds = (edge + 1) / win_prob if win_prob > 0 else 2.0
        b = avg_odds - 1  # Net odds
        
        # Full Kelly fraction
        p = win_prob
        q = 1 - p
        full_kelly = (b * p - q) / b if b > 0 else 0
        
        # Actual stake fraction
        stake = full_kelly * kelly_fraction
        stake = max(0.001, min(stake, 0.25))  # Bound stake
        
        # Log growth rate per bet (Kelly criterion metric)
        # g = p * log(1 + stake * b) + q * log(1 - stake)
        try:
            g = p * np.log(1 + stake * b) + q * np.log(1 - stake)
        except (ValueError, RuntimeWarning):
            g = 0
        
        if g <= 0:
            # Negative growth = ruin is likely
            return min(1.0, 0.5 + abs(g) * n_bets)
        
        # For positive growth, ruin probability decreases exponentially
        # Simplified approximation
        ruin_prob = np.exp(-2 * g * n_bets * np.log(1 / ruin_threshold))
        
        return min(1.0, max(0.0, ruin_prob))
    
    def monte_carlo_drawdown(
        self,
        bet_log: Optional[List[BetRecord]] = None,
        n_simulations: int = 10000,
        n_bets: int = 500,
        edge: float = 0.05,
        win_prob: float = 0.55,
        avg_odds: float = 2.0,
        kelly_fraction: float = 0.25,
        max_stake_pct: float = 0.01,
        ruin_threshold: float = 0.1,
        seed: Optional[int] = None
    ) -> DrawdownStats:
        """
        Monte Carlo simulation of bankroll evolution and drawdowns.
        
        Either uses historical bet log (resampling) or generates
        synthetic bets from parameters.
        
        Args:
            bet_log: Historical bets to resample (if None, use synthetic)
            n_simulations: Number of Monte Carlo simulations
            n_bets: Bets per simulation path
            edge: Expected edge (for synthetic)
            win_prob: Win probability (for synthetic)
            avg_odds: Average odds (for synthetic)
            kelly_fraction: Kelly fraction
            max_stake_pct: Max stake per bet
            ruin_threshold: Fraction considered ruin
            seed: Random seed for reproducibility
            
        Returns:
            DrawdownStats with simulation results
        """
        if seed is not None:
            np.random.seed(seed)
        
        max_drawdowns = []
        ruin_count = 0
        
        for _ in range(n_simulations):
            # Simulate one path
            bankroll = 1.0
            peak = 1.0
            path_max_dd = 0.0
            
            for _ in range(n_bets):
                if bankroll <= ruin_threshold:
                    ruin_count += 1
                    path_max_dd = 1.0 - ruin_threshold
                    break
                
                # Generate or sample bet
                if bet_log:
                    bet = np.random.choice(bet_log)
                    stake_pct = bet.stake_pct
                    odds = bet.odds
                    won = np.random.random() < bet.edge / (odds - 1) + 1 / odds
                else:
                    # Synthetic bet
                    stake_pct = self._kelly_stake(
                        win_prob, avg_odds, kelly_fraction, max_stake_pct
                    )
                    odds = avg_odds
                    won = np.random.random() < win_prob
                
                # Apply bet result
                stake = stake_pct * bankroll
                if won:
                    bankroll += stake * (odds - 1)
                else:
                    bankroll -= stake
                
                # Track drawdown
                if bankroll > peak:
                    peak = bankroll
                else:
                    dd = (peak - bankroll) / peak
                    path_max_dd = max(path_max_dd, dd)
            
            max_drawdowns.append(path_max_dd)
        
        max_drawdowns = np.array(max_drawdowns)
        
        return DrawdownStats(
            mean_max_drawdown=float(np.mean(max_drawdowns)),
            median_max_drawdown=float(np.median(max_drawdowns)),
            percentile_95=float(np.percentile(max_drawdowns, 95)),
            percentile_99=float(np.percentile(max_drawdowns, 99)),
            worst_drawdown=float(np.max(max_drawdowns)),
            ruin_count=ruin_count,
            ruin_probability=ruin_count / n_simulations,
            simulations=n_simulations
        )
    
    def _kelly_stake(
        self,
        prob: float,
        odds: float,
        kelly_fraction: float,
        max_stake: float
    ) -> float:
        """Calculate Kelly stake fraction."""
        b = odds - 1
        p = prob
        q = 1 - p
        
        full_kelly = (b * p - q) / b if b > 0 else 0
        stake = full_kelly * kelly_fraction
        stake = max(0, min(stake, max_stake))
        
        return stake
    
    def analyze_bet_history(
        self,
        bet_log: List[BetRecord],
        n_simulations: int = 5000
    ) -> dict:
        """
        Comprehensive risk analysis from historical bets.
        
        Args:
            bet_log: Historical bet records
            n_simulations: Monte Carlo simulations
            
        Returns:
            Dict with risk metrics
        """
        if not bet_log:
            return {"error": "No bet history"}
        
        # Calculate historical stats
        edges = [b.edge for b in bet_log]
        win_rate = sum(1 for b in bet_log if b.won) / len(bet_log)
        avg_edge = np.mean(edges)
        avg_odds = np.mean([b.odds for b in bet_log])
        
        # Monte Carlo with resampling
        mc_stats = self.monte_carlo_drawdown(
            bet_log=bet_log,
            n_simulations=n_simulations,
            n_bets=len(bet_log) * 2  # Project forward
        )
        
        # Analytical ruin estimate
        ruin_prob = self.ruin_probability(
            edge=avg_edge,
            win_prob=win_rate,
            kelly_fraction=0.25,
            n_bets=len(bet_log) * 2
        )
        
        return {
            "historical": {
                "total_bets": len(bet_log),
                "win_rate": win_rate,
                "avg_edge": avg_edge,
                "avg_odds": avg_odds,
            },
            "monte_carlo": {
                "mean_max_drawdown": mc_stats.mean_max_drawdown,
                "percentile_95_drawdown": mc_stats.percentile_95,
                "simulated_ruin_prob": mc_stats.ruin_probability,
            },
            "analytical": {
                "ruin_probability": ruin_prob,
            }
        }
