"""
Bankroll management with Kelly criterion stake sizing.
"""
import polars as pl
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BankrollManager:
    """
    Bankroll management with Kelly criterion stake sizing.
    Uses fractional Kelly (1/4 by default) for safety.
    """
    
    initial_bankroll: float = 1000.0
    kelly_fraction: float = 0.25  # 1/4 Kelly
    max_stake_pct: float = 0.03   # Max 3% per bet
    min_stake_pct: float = 0.005  # Min 0.5% per bet
    min_odds: float = 1.20
    max_odds: float = 5.00
    
    def __post_init__(self):
        self.current_bankroll = self.initial_bankroll
        self.bet_history = []
    
    def kelly_stake(self, model_prob: float, odds: float) -> float:
        """
        Calculate optimal stake using Kelly criterion.
        
        Kelly formula: f* = (bp - q) / b
        where:
            b = odds - 1 (net odds)
            p = probability of winning
            q = 1 - p
            
        Args:
            model_prob: Model's win probability
            odds: Decimal odds
            
        Returns:
            Optimal stake as fraction of bankroll
        """
        # Basic validation
        if odds < self.min_odds or odds > self.max_odds:
            return 0.0
        
        if model_prob <= 0 or model_prob >= 1:
            return 0.0
        
        b = odds - 1  # Net odds
        p = model_prob
        q = 1 - p
        
        # Full Kelly
        full_kelly = (b * p - q) / b
        
        # Fractional Kelly for safety
        stake = full_kelly * self.kelly_fraction
        
        # Apply limits
        stake = max(0, stake)  # No negative stakes
        stake = min(stake, self.max_stake_pct)  # Cap at max
        
        # Minimum threshold
        if stake < self.min_stake_pct:
            return 0.0
        
        return stake
    
    def calculate_stake_amount(self, model_prob: float, odds: float) -> float:
        """
        Calculate actual stake amount in currency.
        
        Args:
            model_prob: Model's win probability
            odds: Decimal odds
            
        Returns:
            Stake amount in currency units
        """
        stake_pct = self.kelly_stake(model_prob, odds)
        return round(self.current_bankroll * stake_pct, 2)
    
    def expected_value(self, model_prob: float, odds: float) -> float:
        """
        Calculate expected value per unit bet.
        
        EV = p * (odds - 1) - (1 - p)
        
        Returns:
            Expected profit per unit staked
        """
        return model_prob * (odds - 1) - (1 - model_prob)
    
    def edge(self, model_prob: float, odds: float) -> float:
        """
        Calculate edge (model prob - implied prob).
        
        Returns:
            Edge as decimal (e.g., 0.05 = 5% edge)
        """
        implied_prob = 1 / odds
        return model_prob - implied_prob
    
    def place_bet(self, stake: float, odds: float, won: bool) -> float:
        """
        Record a bet result and update bankroll.
        
        Args:
            stake: Amount staked
            odds: Decimal odds
            won: Whether bet won
            
        Returns:
            Profit/loss from this bet
        """
        if won:
            profit = stake * (odds - 1)
        else:
            profit = -stake
        
        self.current_bankroll += profit
        
        self.bet_history.append({
            "stake": stake,
            "odds": odds,
            "won": won,
            "profit": profit,
            "bankroll_after": self.current_bankroll,
        })
        
        return profit
    
    def get_stats(self) -> dict:
        """Get bankroll statistics."""
        if not self.bet_history:
            return {
                "total_bets": 0,
                "current_bankroll": self.current_bankroll,
                "initial_bankroll": self.initial_bankroll,
            }
        
        total_profit = sum(b["profit"] for b in self.bet_history)
        wins = sum(1 for b in self.bet_history if b["won"])
        total = len(self.bet_history)
        total_staked = sum(b["stake"] for b in self.bet_history)
        
        return {
            "total_bets": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": wins / total if total > 0 else 0,
            "total_profit": total_profit,
            "roi": total_profit / total_staked if total_staked > 0 else 0,
            "current_bankroll": self.current_bankroll,
            "initial_bankroll": self.initial_bankroll,
            "growth": (self.current_bankroll - self.initial_bankroll) / self.initial_bankroll,
        }
    
    def add_stakes_to_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add stake calculations to a DataFrame of predictions.
        
        Expects columns: model_prob, odds_player
        """
        stakes = []
        for row in df.iter_rows(named=True):
            prob = row.get("model_prob", 0)
            odds = row.get("odds_player", 0) or 0
            stake_pct = self.kelly_stake(prob, odds)
            stakes.append(stake_pct)
        
        return df.with_columns([
            pl.Series("stake_pct", stakes),
            pl.Series("stake_amount", [s * self.current_bankroll for s in stakes]),
        ])
