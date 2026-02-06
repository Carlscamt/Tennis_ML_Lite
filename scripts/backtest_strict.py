"""
Strict Walk-Forward Backtesting Framework.

This script implements an expanding window backtest to prevent temporal data leakage.
It simulates the passage of time by retraining the model iteratively.

Methodology:
1. Define a START_DATE (e.g., 2024-01-01).
2. For each month M from START_DATE to PRESENT:
   a. Train Model on all data where timestamp < M_start_timestamp.
   b. Predict matches occurring in month M.
   c. Calculate betting performance for month M.
   d. Add month M to training set for next iteration.

Usage:
    python scripts/backtest_strict.py --start-date 2024-01-01
"""
import sys
from pathlib import Path
import warnings
from datetime import datetime, timedelta
import argparse
import logging
from typing import List, Dict, Tuple

import polars as pl
import numpy as np
from dateutil.relativedelta import relativedelta

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import PROCESSED_DATA_DIR, BACKTEST_DIR, BETTING
from src.model.trainer import ModelTrainer
from src.transform.features import FeatureEngineer
from src.betting import BankrollManager, ValueBetFinder
from src.utils import setup_logging

# Suppress XGBoost warnings
warnings.filterwarnings("ignore")

logger = setup_logging()

class WalkForwardBacktester:
    def __init__(self, data_path: Path, start_date: str, bankroll: float = 1000.0):
        self.data_path = data_path
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.initial_bankroll = bankroll
        self.results = []
        self.feature_engineer = FeatureEngineer()
        
    def load_data(self) -> pl.DataFrame:
        """Load and prepare data."""
        logger.info(f"Loading data from {self.data_path}")
        df = pl.read_parquet(self.data_path)
        
        # Ensure strict sorting by time
        df = df.sort("start_timestamp")
        
        # Filter rows where we have outcomes
        df = df.filter(pl.col("player_won").is_not_null())
        
        logger.info(f"Loaded {len(df):,} valid matches")
        return df

    def get_monthly_windows(self, df: pl.DataFrame) -> List[Tuple[datetime, datetime]]:
        """Generate (start, end) tuples for each month in the backtest."""
        # Get dataset range
        last_ts = df["start_timestamp"].max()
        last_date = datetime.fromtimestamp(last_ts)
        
        current = self.start_date
        windows = []
        
        while current < last_date:
            next_month = current + relativedelta(months=1)
            windows.append((current, next_month))
            current = next_month
            
        return windows

    def run(self):
        """Execute the walk-forward backtest."""
        df = self.load_data()
        
        # Get features safely (using the logic from FeatureEngineer to avoid leakage)
        # We need a dummy lazyframe to get columns
        feature_cols = self.feature_engineer.get_feature_columns(df.lazy())
        logger.info(f"Using {len(feature_cols)} features for training")
        
        windows = self.get_monthly_windows(df)
        logger.info(f"Starting Walk-Forward Validation over {len(windows)} months")
        
        all_blind_predictions = []
        
        # Initialize Bankroll Manager for the entire simulation
        bankroll = BankrollManager(initial_bankroll=self.initial_bankroll)
        value_finder = ValueBetFinder(
            min_edge=BETTING.min_edge,
            min_confidence=BETTING.min_confidence
        )
        
        total_months = len(windows)
        
        for i, (month_start, month_end) in enumerate(windows):
            month_start_ts = month_start.timestamp()
            month_end_ts = month_end.timestamp()
            
            # 1. SPLIT DATA
            # Train set: All history BEFORE this month
            train_df = df.filter(pl.col("start_timestamp") < month_start_ts)
            
            # Test set: Matches IN this month
            test_df = df.filter(
                (pl.col("start_timestamp") >= month_start_ts) &
                (pl.col("start_timestamp") < month_end_ts)
            ).filter(pl.col("odds_player").is_not_null())
            
            if len(test_df) == 0:
                logger.warning(f"Month {month_start.strftime('%Y-%m')} empty. Skipping.")
                continue
            
            # 2. TRAIN MODEL (Retrain from scratch to prevent leakage)
            # We use a smaller model for speed in backtesting, or match production params
            trainer = ModelTrainer(calibrate=True) 
            
            # Train
            logger.info(f"[{i+1}/{total_months}] {month_start.strftime('%Y-%m')}: Training on {len(train_df):,} | Testing on {len(test_df):,}")
            trainer.train(train_df, feature_cols, target_col="player_won")
            
            # 3. PREDICT
            probs = trainer.predict_proba(test_df)
            
            # Store predictions with metadata
            test_df = test_df.with_columns(
                pl.Series("model_prob", probs)
            )
            
            # 4. SIMULATE BETTING DAY-BY-DAY (within the month)
            # We iterate unique dates to apply "max_bets_per_day" correctly
            month_profit = 0
            bets_placed = 0
            
            # Get unique dates in this month
            unique_dates = test_df["match_date"].unique().sort()
            
            for date in unique_dates:
                # Get matches for this day
                daily_df = test_df.filter(pl.col("match_date") == date)
                
                # Find value bets for this day
                # find_value_bets returns ONLY the bets to place (already filtered)
                daily_value_bets = value_finder.find_value_bets(daily_df)
                
                for row in daily_value_bets.iter_rows(named=True):
                    stake = bankroll.calculate_stake_amount(row["model_prob"], row["odds_player"])
                    if stake > 0:
                        profit = bankroll.place_bet(stake, row["odds_player"], row["player_won"])
                        month_profit += profit
                        bets_placed += 1
            
            # Log monthly result
            roi = (month_profit / bankroll.current_bankroll) if bankroll.current_bankroll > 0 else 0
            logger.info(f"   -> Result: {bets_placed} bets, Profit: ${month_profit:.2f}, Bankroll: ${bankroll.current_bankroll:.0f}")
            
            # Append to master list
            all_blind_predictions.append(test_df)
            
        # CONSOLIDATE RESULTS
        if not all_blind_predictions:
            logger.error("No predictions made during backtest.")
            return

        full_results = pl.concat(all_blind_predictions)
        
        # Save raw results
        BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
        save_path = BACKTEST_DIR / f"strict_backtest_{self.start_date.strftime('%Y%m%d')}.parquet"
        full_results.write_parquet(save_path)
        
        # Final Stats
        stats = bankroll.get_stats()
        logger.info("=" * 60)
        logger.info("STRICT BACKTEST FINAL REPORT")
        logger.info("=" * 60)
        logger.info(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}")
        logger.info(f"Total Bets: {stats['total_bets']}")
        logger.info(f"Win Rate:   {stats['win_rate']:.1%}")
        logger.info(f"ROI:        {stats['roi']:.2%}")
        logger.info(f"Profit:     ${stats['total_profit']:.2f}")
        logger.info(f"Final Bank: ${stats['current_bankroll']:.2f}")
        logger.info(f"Calibration Error (ECE): {self._calculate_ece(full_results):.4f}")
        logger.info("=" * 60)

    def _calculate_ece(self, df: pl.DataFrame, n_bins=10) -> float:
        """Calculate Expected Calibration Error."""
        df = df.select(["model_prob", "player_won"]).drop_nulls()
        if len(df) == 0: return 0.0
        
        probs = df["model_prob"].to_numpy()
        actual = df["player_won"].to_numpy()
        
        # Binning
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_prob = np.mean(probs[mask])
                bin_acc = np.mean(actual[mask])
                weight = np.sum(mask) / len(probs)
                ece += weight * np.abs(bin_prob - bin_acc)
                
        return ece

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Walk-Forward Backtester")
    parser.add_argument("--start-date", type=str, required=True, help="YYYY-MM-DD start date")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    
    args = parser.parse_args()
    
    data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        sys.exit(1)
        
    backtester = WalkForwardBacktester(data_path, args.start_date, args.bankroll)
    backtester.run()
