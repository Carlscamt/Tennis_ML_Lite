"""
E2E tests for bankroll simulation.

Tests full pipeline with betting logic:
- Bankroll evolution over historical period
- Max drawdown assertions
- Stake sizing cap assertions
"""
import pytest
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from typing import List
import polars as pl
import numpy as np


@dataclass
class BetResult:
    """Single bet result."""
    event_id: int
    stake: float
    odds: float
    predicted_prob: float
    won: bool
    pnl: float


@dataclass
class SimulationConfig:
    """Simulation configuration."""
    initial_bankroll: float = 1000.0
    max_stake_pct: float = 0.05       # Max 5% of bankroll per bet
    min_edge: float = 0.05            # Minimum edge to bet
    max_drawdown_pct: float = 0.25    # Max 25% drawdown allowed
    kelly_fraction: float = 0.25      # Quarter Kelly


class BankrollSimulator:
    """Simulate bankroll evolution over time."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.bankroll = config.initial_bankroll
        self.peak_bankroll = config.initial_bankroll
        self.bets: List[BetResult] = []
        self.bankroll_history: List[float] = [config.initial_bankroll]
    
    def calculate_stake(self, prob: float, odds: float) -> float:
        """Calculate Kelly stake."""
        edge = (prob * odds) - 1
        if edge <= 0:
            return 0.0
        
        # Kelly fraction
        q = 1 - prob
        b = odds - 1
        kelly = (b * prob - q) / b if b > 0 else 0
        
        # Apply fraction and cap
        stake_pct = kelly * self.config.kelly_fraction
        stake_pct = min(stake_pct, self.config.max_stake_pct)
        
        return stake_pct * self.bankroll
    
    def place_bet(self, event_id: int, prob: float, odds: float, won: bool) -> BetResult:
        """Place a bet and update bankroll."""
        stake = self.calculate_stake(prob, odds)
        
        if stake <= 0:
            return None
        
        if won:
            pnl = stake * (odds - 1)
        else:
            pnl = -stake
        
        self.bankroll += pnl
        self.bankroll_history.append(self.bankroll)
        
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        
        result = BetResult(
            event_id=event_id,
            stake=stake,
            odds=odds,
            predicted_prob=prob,
            won=won,
            pnl=pnl,
        )
        self.bets.append(result)
        return result
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_bankroll == 0:
            return 0.0
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll
    
    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown experienced."""
        if not self.bankroll_history:
            return 0.0
        
        peak = self.bankroll_history[0]
        max_dd = 0.0
        
        for value in self.bankroll_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    @property
    def total_pnl(self) -> float:
        """Total profit/loss."""
        return self.bankroll - self.config.initial_bankroll
    
    @property
    def roi(self) -> float:
        """Return on investment."""
        total_staked = sum(b.stake for b in self.bets)
        if total_staked == 0:
            return 0.0
        return self.total_pnl / total_staked


@pytest.mark.e2e
class TestBankrollSimulation:
    """E2E tests for bankroll simulation."""
    
    @pytest.fixture
    def config(self):
        """Default simulation config."""
        return SimulationConfig(
            initial_bankroll=1000.0,
            max_stake_pct=0.05,
            min_edge=0.05,
            max_drawdown_pct=0.25,
        )
    
    @pytest.fixture
    def historical_bets(self):
        """Sample historical bet data."""
        np.random.seed(42)
        n_bets = 50
        
        return pl.DataFrame({
            "event_id": range(n_bets),
            "predicted_prob": np.random.uniform(0.4, 0.8, n_bets),
            "odds": np.random.uniform(1.3, 3.0, n_bets),
            "actual_won": np.random.choice([True, False], n_bets, p=[0.55, 0.45]),
        })
    
    def test_full_simulation_runs(self, config, historical_bets):
        """Full simulation runs without errors."""
        sim = BankrollSimulator(config)
        
        for row in historical_bets.iter_rows(named=True):
            edge = (row["predicted_prob"] * row["odds"]) - 1
            if edge >= config.min_edge:
                sim.place_bet(
                    event_id=row["event_id"],
                    prob=row["predicted_prob"],
                    odds=row["odds"],
                    won=row["actual_won"],
                )
        
        # Should have placed some bets
        assert len(sim.bets) > 0
        
        # Bankroll should be positive
        assert sim.bankroll > 0
    
    def test_stake_never_exceeds_cap(self, config, historical_bets):
        """No stake exceeds configured cap."""
        sim = BankrollSimulator(config)
        
        for row in historical_bets.iter_rows(named=True):
            edge = (row["predicted_prob"] * row["odds"]) - 1
            if edge >= config.min_edge:
                # Record bankroll before bet
                bankroll_before = sim.bankroll
                max_allowed = bankroll_before * config.max_stake_pct
                
                result = sim.place_bet(
                    event_id=row["event_id"],
                    prob=row["predicted_prob"],
                    odds=row["odds"],
                    won=row["actual_won"],
                )
                
                if result:
                    assert result.stake <= max_allowed * 1.001, (
                        f"Stake {result.stake:.2f} exceeds cap {max_allowed:.2f}"
                    )
    
    def test_max_drawdown_within_limit(self, config, historical_bets):
        """Max drawdown stays within configured limit."""
        sim = BankrollSimulator(config)
        
        for row in historical_bets.iter_rows(named=True):
            edge = (row["predicted_prob"] * row["odds"]) - 1
            if edge >= config.min_edge:
                sim.place_bet(
                    event_id=row["event_id"],
                    prob=row["predicted_prob"],
                    odds=row["odds"],
                    won=row["actual_won"],
                )
        
        # Check max drawdown
        assert sim.max_drawdown < config.max_drawdown_pct, (
            f"Max drawdown {sim.max_drawdown:.2%} exceeds limit {config.max_drawdown_pct:.2%}"
        )
    
    def test_no_negative_bankroll(self, config, historical_bets):
        """Bankroll never goes negative."""
        sim = BankrollSimulator(config)
        
        for row in historical_bets.iter_rows(named=True):
            edge = (row["predicted_prob"] * row["odds"]) - 1
            if edge >= config.min_edge:
                sim.place_bet(
                    event_id=row["event_id"],
                    prob=row["predicted_prob"],
                    odds=row["odds"],
                    won=row["actual_won"],
                )
                
                assert sim.bankroll >= 0, "Bankroll went negative"
    
    def test_bankroll_history_tracked(self, config, historical_bets):
        """Bankroll history is correctly tracked."""
        sim = BankrollSimulator(config)
        
        for row in historical_bets.iter_rows(named=True):
            edge = (row["predicted_prob"] * row["odds"]) - 1
            if edge >= config.min_edge:
                sim.place_bet(
                    event_id=row["event_id"],
                    prob=row["predicted_prob"],
                    odds=row["odds"],
                    won=row["actual_won"],
                )
        
        # History length = initial + number of bets
        assert len(sim.bankroll_history) == len(sim.bets) + 1
        
        # Final value matches current
        assert sim.bankroll_history[-1] == sim.bankroll


@pytest.mark.e2e
class TestStakeSizingRules:
    """E2E tests for stake sizing business rules."""
    
    @pytest.fixture
    def aggressive_config(self):
        """Aggressive config for edge cases."""
        return SimulationConfig(
            initial_bankroll=1000.0,
            max_stake_pct=0.10,  # 10% max
            kelly_fraction=0.5,  # Half Kelly
        )
    
    def test_kelly_fraction_applied(self, aggressive_config):
        """Kelly fraction reduces stake from full Kelly."""
        sim = BankrollSimulator(aggressive_config)
        
        # High edge bet
        prob = 0.7
        odds = 2.0
        
        stake = sim.calculate_stake(prob, odds)
        
        # Full Kelly would be (0.7 * 2 - 0.3) / 1 = 1.1 = 110%
        # Half Kelly = 55%, capped at 10%
        assert stake <= aggressive_config.max_stake_pct * sim.bankroll
    
    def test_no_bet_on_negative_edge(self, aggressive_config):
        """No stake for negative edge bets."""
        sim = BankrollSimulator(aggressive_config)
        
        # Negative edge
        prob = 0.3
        odds = 2.0  # Edge = 0.3 * 2 - 1 = -0.4
        
        stake = sim.calculate_stake(prob, odds)
        
        assert stake == 0.0


@pytest.mark.e2e
class TestDrawdownProtection:
    """E2E tests for drawdown protection."""
    
    def test_losing_streak_respects_limits(self):
        """Extended losing streak still respects stake limits."""
        config = SimulationConfig(
            initial_bankroll=1000.0,
            max_stake_pct=0.05,
        )
        sim = BankrollSimulator(config)
        
        # 20 consecutive losses
        for i in range(20):
            sim.place_bet(
                event_id=i,
                prob=0.6,
                odds=2.0,
                won=False,
            )
        
        # Bankroll should still be positive
        assert sim.bankroll > 0, "Bankroll depleted during losing streak"
        
        # Max drawdown should be calculable
        assert sim.max_drawdown > 0
        assert sim.max_drawdown < 1.0  # Not total loss
