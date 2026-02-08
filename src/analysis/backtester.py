"""
Enhanced backtesting with realistic simulation and analytics.

Features:
- Latency simulation (delay between prediction and bet execution)
- Book selection rules (best, average, specific book)
- Daily/monthly ROI, volatility, and drawdown aggregation
- Kelly fraction and edge threshold tuning
"""
import polars as pl
import numpy as np
from typing import Optional, List, Dict, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for enhanced backtesting."""
    
    # Latency simulation
    latency_minutes: int = 5  # Delay between prediction and bet
    line_movement_pct: float = 0.02  # Expected line movement (2%)
    
    # Book selection
    book_selection: Literal["best", "average", "specific"] = "average"
    specific_book: Optional[str] = None  # For "specific" mode
    
    # Betting parameters
    kelly_fraction: float = 0.25
    min_edge: float = 0.05
    max_bet_fraction: float = 0.01
    initial_bankroll: float = 1000.0
    
    # Filters
    min_confidence: float = 0.55
    min_odds: float = 1.30
    max_odds: float = 5.00
    
    # Tournament filters
    exclude_tiers: List[str] = field(default_factory=list)  # e.g., ["futures"]


@dataclass 
class BacktestResult:
    """Results from backtest simulation."""
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    total_profit: float
    total_staked: float
    roi_pct: float
    final_bankroll: float
    max_drawdown: float
    sharpe_ratio: float
    daily_stats: pl.DataFrame
    monthly_stats: pl.DataFrame
    bet_log: pl.DataFrame


class EnhancedBacktester:
    """
    Backtester with realistic simulation and comprehensive analytics.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
    
    def run(self, predictions_df: pl.DataFrame) -> BacktestResult:
        """
        Run backtest on predictions DataFrame.
        
        Expected columns:
        - model_prob, odds_player, odds_opponent
        - player_won (outcome)
        - start_timestamp
        - Optional: tournament_tier, odds_best, odds_avg
        """
        df = predictions_df.clone()
        
        # Apply filters
        df = self._apply_filters(df)
        
        # Apply latency/line movement simulation
        df = self._simulate_latency(df)
        
        # Select book odds
        df = self._select_book_odds(df)
        
        # Calculate edge with executed odds
        df = self._calculate_edge(df)
        
        # Filter by edge threshold
        df = df.filter(pl.col("edge") >= self.config.min_edge)
        
        # Sort by timestamp for sequential simulation
        df = df.sort("start_timestamp")
        
        # Simulate betting with Kelly sizing
        bet_log = self._simulate_betting(df)
        
        # Calculate aggregate statistics
        result = self._calculate_statistics(bet_log)
        
        return result
    
    def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply base filters."""
        filters = (
            (pl.col("model_prob") >= self.config.min_confidence) &
            (pl.col("odds_player") >= self.config.min_odds) &
            (pl.col("odds_player") <= self.config.max_odds) &
            (pl.col("player_won").is_not_null())
        )
        
        # Exclude tournament tiers if specified
        if self.config.exclude_tiers and "tournament_tier" in df.columns:
            filters = filters & (~pl.col("tournament_tier").is_in(self.config.exclude_tiers))
        
        return df.filter(filters)
    
    def _simulate_latency(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Simulate line movement due to latency.
        
        Assumes odds move against you by line_movement_pct on average.
        """
        if self.config.latency_minutes == 0:
            return df.with_columns([
                pl.col("odds_player").alias("executed_odds")
            ])
        
        # Scale movement by latency (5 min = base, 30 min = 6x)
        latency_factor = self.config.latency_minutes / 5.0
        movement = self.config.line_movement_pct * latency_factor
        
        # Odds move against us (decrease for favorites, increase for underdogs)
        # Simplified: reduce all odds slightly (less favorable)
        return df.with_columns([
            (pl.col("odds_player") * (1 - movement)).clip(1.01, 100).alias("executed_odds")
        ])
    
    def _select_book_odds(self, df: pl.DataFrame) -> pl.DataFrame:
        """Select odds based on book selection strategy."""
        if self.config.book_selection == "best" and "odds_best" in df.columns:
            return df.with_columns([
                pl.col("odds_best").alias("bet_odds")
            ])
        elif self.config.book_selection == "average" and "odds_avg" in df.columns:
            return df.with_columns([
                pl.col("odds_avg").alias("bet_odds")
            ])
        else:
            # Default: use executed_odds (with latency applied)
            return df.with_columns([
                pl.col("executed_odds").alias("bet_odds")
            ])
    
    def _calculate_edge(self, df: pl.DataFrame) -> pl.DataFrame:
        """Recalculate edge with executed odds."""
        return df.with_columns([
            (1 / pl.col("bet_odds")).alias("implied_prob"),
            (pl.col("model_prob") - 1 / pl.col("bet_odds")).alias("edge"),
            (
                pl.col("model_prob") * (pl.col("bet_odds") - 1) -
                (1 - pl.col("model_prob"))
            ).alias("expected_value")
        ])
    
    def _simulate_betting(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Simulate sequential betting with Kelly sizing.
        """
        bankroll = self.config.initial_bankroll
        bet_records = []
        
        for row in df.iter_rows(named=True):
            prob = row["model_prob"]
            odds = row["bet_odds"]
            won = row["player_won"]
            
            # Kelly stake calculation
            b = odds - 1
            full_kelly = (b * prob - (1 - prob)) / b
            stake_pct = max(0, full_kelly * self.config.kelly_fraction)
            stake_pct = min(stake_pct, self.config.max_bet_fraction)
            
            if stake_pct < 0.005:  # Min stake threshold
                continue
            
            stake = bankroll * stake_pct
            
            # Calculate P&L
            if won:
                pnl = stake * (odds - 1)
            else:
                pnl = -stake
            
            bankroll += pnl
            
            bet_records.append({
                "timestamp": row.get("start_timestamp", 0),
                "odds": odds,
                "model_prob": prob,
                "edge": row["edge"],
                "stake": stake,
                "stake_pct": stake_pct,
                "won": won,
                "pnl": pnl,
                "bankroll": bankroll
            })
        
        if not bet_records:
            return pl.DataFrame()
        
        return pl.DataFrame(bet_records)
    
    def _calculate_statistics(self, bet_log: pl.DataFrame) -> BacktestResult:
        """Calculate comprehensive statistics."""
        if bet_log.is_empty():
            return BacktestResult(
                total_bets=0, wins=0, losses=0, win_rate=0,
                total_profit=0, total_staked=0, roi_pct=0,
                final_bankroll=self.config.initial_bankroll,
                max_drawdown=0, sharpe_ratio=0,
                daily_stats=pl.DataFrame(),
                monthly_stats=pl.DataFrame(),
                bet_log=bet_log
            )
        
        # Basic stats
        total_bets = len(bet_log)
        wins = bet_log.filter(pl.col("won")).height
        losses = total_bets - wins
        total_profit = bet_log["pnl"].sum()
        total_staked = bet_log["stake"].sum()
        final_bankroll = bet_log["bankroll"].tail(1)[0]
        
        # Max drawdown
        cummax = bet_log["bankroll"].cum_max()
        drawdowns = (cummax - bet_log["bankroll"]) / cummax
        max_drawdown = drawdowns.max()
        
        # Sharpe ratio (daily returns)
        bet_log = bet_log.with_columns([
            pl.from_epoch("timestamp", time_unit="s").alias("datetime")
        ])
        
        daily_returns = bet_log.with_columns([
            pl.col("datetime").dt.date().alias("date")
        ]).group_by("date").agg([
            pl.col("pnl").sum().alias("daily_pnl"),
            pl.col("stake").sum().alias("daily_staked"),
            pl.len().alias("bets"),
            pl.col("won").sum().alias("wins")
        ]).with_columns([
            (pl.col("daily_pnl") / pl.col("daily_staked")).alias("daily_return")
        ]).sort("date")
        
        if len(daily_returns) > 1:
            mean_return = daily_returns["daily_return"].mean()
            std_return = daily_returns["daily_return"].std()
            sharpe = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Monthly aggregation
        monthly_stats = bet_log.with_columns([
            pl.col("datetime").dt.strftime("%Y-%m").alias("month")
        ]).group_by("month").agg([
            pl.len().alias("bets"),
            pl.col("won").sum().alias("wins"),
            pl.col("pnl").sum().alias("profit"),
            pl.col("stake").sum().alias("staked"),
        ]).with_columns([
            (pl.col("profit") / pl.col("staked") * 100).alias("roi_pct"),
            (pl.col("wins") / pl.col("bets") * 100).alias("win_rate")
        ]).sort("month")
        
        # Daily stats with cumulative metrics
        daily_stats = daily_returns.with_columns([
            pl.col("daily_pnl").cum_sum().alias("cumulative_profit"),
            (pl.col("daily_pnl").cum_sum() / self.config.initial_bankroll * 100).alias("cumulative_roi_pct")
        ])
        
        return BacktestResult(
            total_bets=total_bets,
            wins=wins,
            losses=losses,
            win_rate=wins / total_bets if total_bets else 0,
            total_profit=total_profit,
            total_staked=total_staked,
            roi_pct=total_profit / total_staked * 100 if total_staked else 0,
            final_bankroll=final_bankroll,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            daily_stats=daily_stats,
            monthly_stats=monthly_stats,
            bet_log=bet_log
        )
    
    def tune_parameters(
        self,
        predictions_df: pl.DataFrame,
        kelly_fractions: List[float] = [0.1, 0.25, 0.5],
        edge_thresholds: List[float] = [0.03, 0.05, 0.07, 0.10],
        latencies: List[int] = [0, 5, 15, 30]
    ) -> pl.DataFrame:
        """
        Grid search over parameter combinations.
        
        Returns DataFrame with results for each combination.
        """
        results = []
        
        for kf in kelly_fractions:
            for edge in edge_thresholds:
                for latency in latencies:
                    config = BacktestConfig(
                        kelly_fraction=kf,
                        min_edge=edge,
                        latency_minutes=latency,
                        initial_bankroll=self.config.initial_bankroll
                    )
                    
                    backtester = EnhancedBacktester(config)
                    result = backtester.run(predictions_df)
                    
                    results.append({
                        "kelly_fraction": kf,
                        "min_edge": edge,
                        "latency_min": latency,
                        "total_bets": result.total_bets,
                        "roi_pct": result.roi_pct,
                        "max_drawdown": result.max_drawdown,
                        "sharpe_ratio": result.sharpe_ratio,
                        "final_bankroll": result.final_bankroll
                    })
        
        return pl.DataFrame(results).sort("roi_pct", descending=True)
    
    def print_report(self, result: BacktestResult) -> None:
        """Print formatted backtest report."""
        print("=" * 60)
        print("ENHANCED BACKTEST REPORT")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Kelly Fraction:  {self.config.kelly_fraction}")
        print(f"  Min Edge:        {self.config.min_edge:.1%}")
        print(f"  Latency:         {self.config.latency_minutes} min")
        print(f"  Book Selection:  {self.config.book_selection}")
        print()
        print(f"Results:")
        print(f"  Total Bets:      {result.total_bets:,}")
        print(f"  Win Rate:        {result.win_rate:.1%}")
        print(f"  Total Profit:    ${result.total_profit:,.2f}")
        print(f"  ROI:             {result.roi_pct:+.1f}%")
        print(f"  Final Bankroll:  ${result.final_bankroll:,.2f}")
        print(f"  Max Drawdown:    {result.max_drawdown:.1%}")
        print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print()
        
        if not result.monthly_stats.is_empty():
            print("Monthly Performance:")
            print(f"{'Month':<10} | {'Bets':>5} | {'Win%':>6} | {'Profit':>10} | {'ROI':>8}")
            print("-" * 50)
            for row in result.monthly_stats.iter_rows(named=True):
                print(f"{row['month']:<10} | {row['bets']:>5} | {row['win_rate']:>5.1f}% | ${row['profit']:>+8.2f} | {row['roi_pct']:>+7.1f}%")
