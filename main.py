#!/usr/bin/env python
"""
Tennis ML Lite - Main CLI
"""
import argparse
import polars as pl
from config import DATA_DIR, MIN_ODDS, MAX_ODDS, MIN_EDGE

def cmd_scrape(args):
    from scraper import scrape_upcoming
    scrape_upcoming(days=args.days)

def cmd_train(args):
    from model import train
    from features import add_features
    
    # Load historical data
    data_file = DATA_DIR / "historical.parquet"
    if not data_file.exists():
        print("No historical data. Scrape first: python main.py scrape")
        return
    
    df = pl.read_parquet(data_file)
    df = add_features(df)
    train(df)

def cmd_predict(args):
    from model import predict
    from features import compute_prediction_features
    from scraper import scrape_upcoming
    
    # Load or scrape upcoming
    upcoming_file = DATA_DIR / "upcoming.parquet"
    if not upcoming_file.exists() or args.scrape:
        scrape_upcoming(args.days)
    
    upcoming = pl.read_parquet(upcoming_file)
    
    # Load historical for features
    hist_file = DATA_DIR / "historical.parquet"
    if hist_file.exists():
        historical = pl.read_parquet(hist_file)
        upcoming = compute_prediction_features(upcoming, historical)
    
    # Predict
    predictions = predict(upcoming)
    
    # Filter value bets
    value_bets = predictions.filter(
        (pl.col("edge") > args.min_edge) &
        (pl.col("odds_player") >= args.min_odds) &
        (pl.col("odds_player") <= args.max_odds)
    ).sort("edge", descending=True)
    
    # Display
    print(f"\n=== VALUE BETS ({len(value_bets)}) ===\n")
    
    for i, row in enumerate(value_bets.head(10).iter_rows(named=True), 1):
        print(f"#{i} >>> BET ON: {row['player_name']}")
        print(f"    vs {row['opponent_name']}")
        print(f"    Prob: {row['model_prob']*100:.1f}% | Odds: {row['odds_player']:.2f} | Edge: +{row['edge']*100:.1f}%")
        print(f"    {row.get('tournament', '')}")
        print()

def main():
    parser = argparse.ArgumentParser(description="Tennis ML Lite")
    subs = parser.add_subparsers(dest="cmd", required=True)
    
    # Scrape
    p = subs.add_parser("scrape")
    p.add_argument("--days", type=int, default=7)
    p.set_defaults(func=cmd_scrape)
    
    # Train
    p = subs.add_parser("train")
    p.set_defaults(func=cmd_train)
    
    # Predict
    p = subs.add_parser("predict")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--min-odds", type=float, default=MIN_ODDS)
    p.add_argument("--max-odds", type=float, default=MAX_ODDS)
    p.add_argument("--min-edge", type=float, default=MIN_EDGE)
    p.add_argument("--scrape", action="store_true")
    p.set_defaults(func=cmd_predict)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
