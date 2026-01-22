#!/usr/bin/env python
"""
Tennis Prediction System - Unified CLI

Single entry point for all tennis prediction workflows.

Usage:
    python tennis.py scrape historical --top 50
    python tennis.py scrape upcoming --days 7
    python tennis.py train
    python tennis.py predict --days 7
    python tennis.py audit
    python tennis.py backtest
"""
import sys
from pathlib import Path
import argparse
import logging

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def cmd_scrape(args):
    """Run scraper with specified mode."""
    from src.scraper import scrape_historical, scrape_upcoming, scrape_players
    
    if args.mode == "historical":
        scrape_historical(
            top_players=args.top,
            max_pages=args.pages,
            resume=args.resume,
            fetch_details=not args.no_details
        )
    elif args.mode == "upcoming":
        scrape_upcoming(days_ahead=args.days)
    elif args.mode == "players":
        if not args.ids:
            print("ERROR: --ids required for players mode")
            return 1
        player_ids = [int(x.strip()) for x in args.ids.split(",")]
        scrape_players(player_ids=player_ids, max_pages=args.pages)


def cmd_train(args):
    """Train or retrain the model."""
    from scripts.run_pipeline import run_data_pipeline, run_training_pipeline
    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
    
    print("=== TRAINING PIPELINE ===")
    
    # Run data pipeline
    print("Step 1: Processing data...")
    run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    
    # Train model
    print("Step 2: Training model...")
    data_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
    run_training_pipeline(data_path, MODELS_DIR)
    
    print("=== TRAINING COMPLETE ===")


def cmd_predict(args):
    """Get predictions for upcoming matches."""
    from src.pipeline import TennisPipeline
    
    pipeline = TennisPipeline()
    predictions = pipeline.predict_upcoming(
        days=args.days,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
        min_confidence=args.confidence,
        scrape_unknown=not args.no_scrape
    )
    
    if len(predictions) == 0:
        print("No predictions available")
        return
    
    # Display top value bets
    print(f"\n=== TOP VALUE BETS ({len(predictions)} matches) ===\n")
    
    # Filter to respect odds range and exclude doubles (names with "/")
    value_bets = predictions.filter(
        (predictions["edge"] > 0.05) &
        (predictions["odds_player"] >= args.min_odds) &
        (predictions["odds_player"] <= args.max_odds) &
        (~predictions["player_name"].str.contains("/")) &
        (~predictions["opponent_name"].str.contains("/"))
    ).sort("edge", descending=True).head(10)
    
    if len(value_bets) == 0:
        print("No value bets found matching criteria.")
        print(f"  (odds range: {args.min_odds}-{args.max_odds}, min edge: 5%)")
        return
    
    print(f"Found {len(value_bets)} value bets:\n")
    
    for i, row in enumerate(value_bets.iter_rows(named=True), 1):
        player = row['player_name']
        opponent = row['opponent_name']
        prob = row.get('model_prob', 0) * 100
        odds = row.get('odds_player', 0)
        edge = row.get('edge', 0) * 100
        tournament = row.get('tournament_name', 'Unknown')
        
        print(f"#{i} >>> BET ON: {player}")
        print(f"    vs {opponent}")
        print(f"    Win Prob: {prob:.1f}% | Odds: {odds:.2f} | Edge: +{edge:.1f}%")
        print(f"    Tournament: {tournament}")
        print()


def cmd_audit(args):
    """Run model audit."""
    from scripts.model_audit import main as run_audit
    run_audit([])


def cmd_backtest(args):
    """Run backtesting."""
    from scripts.backtest import main as run_backtest
    run_backtest([])


def main():
    parser = argparse.ArgumentParser(
        description="Tennis Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tennis.py scrape historical --top 50
  python tennis.py scrape upcoming --days 7
  python tennis.py train
  python tennis.py predict --days 7 --min-odds 1.5 --max-odds 3.0
  python tennis.py audit
  python tennis.py backtest
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # SCRAPE command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape tennis data")
    scrape_parser.add_argument("mode", choices=["historical", "upcoming", "players"])
    scrape_parser.add_argument("--top", type=int, default=50, help="Top N players (historical)")
    scrape_parser.add_argument("--pages", type=int, default=10, help="Max pages per player")
    scrape_parser.add_argument("--days", type=int, default=7, help="Days ahead (upcoming)")
    scrape_parser.add_argument("--ids", help="Player IDs (players mode)")
    scrape_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    scrape_parser.add_argument("--no-details", action="store_true", help="Skip odds/stats")
    scrape_parser.set_defaults(func=cmd_scrape)
    
    # TRAIN command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.set_defaults(func=cmd_train)
    
    # PREDICT command
    predict_parser = subparsers.add_parser("predict", help="Get predictions")
    predict_parser.add_argument("--days", type=int, default=7, help="Days ahead")
    predict_parser.add_argument("--min-odds", type=float, default=1.5, help="Min odds")
    predict_parser.add_argument("--max-odds", type=float, default=3.0, help="Max odds")
    predict_parser.add_argument("--confidence", type=float, default=0.55, help="Min confidence")
    predict_parser.add_argument("--no-scrape", action="store_true", help="Skip auto-scraping unknown players")
    predict_parser.set_defaults(func=cmd_predict)
    
    # AUDIT command
    audit_parser = subparsers.add_parser("audit", help="Run model audit")
    audit_parser.set_defaults(func=cmd_audit)
    
    # BACKTEST command
    backtest_parser = subparsers.add_parser("backtest", help="Run backtesting")
    backtest_parser.set_defaults(func=cmd_backtest)
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
