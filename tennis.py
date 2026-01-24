#!/usr/bin/env python
"""
Tennis Prediction System - Unified CLI with Observability and Model Serving
"""
import sys
from pathlib import Path
import argparse
import os
import time
import uuid
import json
from dataclasses import asdict

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.utils.observability import initialize_observability, get_metrics, Logger, CORRELATION_ID
from src.model.registry import ModelRegistry
from src.model.serving import ServingConfig

# Initialize observability
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
initialize_observability(environment=ENVIRONMENT)

logger = Logger(__name__)
metrics = get_metrics()


def log_metrics_to_stdout():
    """Print metrics for monitoring (development fallback)."""
    if ENVIRONMENT == 'production':
        return

    try:
        from prometheus_client import generate_latest
        metrics_bytes = generate_latest(metrics.registry)
        # Only print non-empty metrics or summary
        pass 
    except Exception:
        pass


def cmd_scrape(args):
    """Run scraper with observability."""
    from src.scraper import scrape_historical, scrape_upcoming, scrape_players
    
    logger.log_event('scrape_command_started', mode=args.mode)
    
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
            logger.log_error("scrape_error_missing_ids")
            print("ERROR: --ids required for players mode")
            return 1
        player_ids = [int(x.strip()) for x in args.ids.split(",")]
        scrape_players(player_ids=player_ids, max_pages=args.pages)


def cmd_train(args):
    """Train pipeline with observability."""
    from src.pipeline import TennisPipeline
    
    logger.log_event('train_command_started')
    pipeline = TennisPipeline()
    
    # Run data pipeline
    print("Step 1: Processing data...")
    data_result = pipeline.run_data_pipeline()
    
    # Run training
    print("Step 2: Training model...")
    pipeline.run_training_pipeline(Path(data_result['output_path']))
    
    print("=== TRAINING COMPLETE ===")
    print("Model registered globally as 'Experimental'. Use 'list-models' to view.")


def cmd_predict(args):
    """Predict with observability."""
    from src.pipeline import TennisPipeline
    
    logger.log_event('predict_command_started', days=args.days)
    
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
    
    # Display logic
    print(f"\n=== TOP VALUE BETS (Limit: {args.limit}) ===\n")
    
    value_bets = predictions.filter(
        (predictions["edge"] > 0.05) &
        (predictions["odds_player"] >= args.min_odds) &
        (predictions["odds_player"] <= args.max_odds) &
        (~predictions["player_name"].str.contains("/")) &
        (~predictions["opponent_name"].str.contains("/"))
    ).sort("edge", descending=True).head(args.limit)
    
    if len(value_bets) == 0:
        print("No value bets found matching criteria.")
        return
    
    print(f"Found {len(value_bets)} value bets:\n")
    
    for i, row in enumerate(value_bets.iter_rows(named=True), 1):
        player = row['player_name']
        opponent = row['opponent_name']
        prob = row.get('model_prob', 0) * 100
        odds = row.get('odds_player', 0)
        edge = row.get('edge', 0) * 100
        tournament = row.get('tournament_name', 'Unknown')
        match_date = row.get('match_date', 'Unknown')
        model_ver = row.get('model_version', 'N/A')
        mode = row.get('serving_mode', 'N/A')
        
        print(f"#{i} >>> BET ON: {player}")
        print(f"    vs {opponent}")
        print(f"    Date: {match_date}")
        print(f"    Win Prob: {prob:.1f}% | Odds: {odds:.2f} | Edge: +{edge:.1f}%")
        print(f"    Tournament: {tournament}")
        print(f"    [Model: {model_ver} | Mode: {mode}]")
        print()


def cmd_audit(args):
    from scripts.model_audit import main as run_audit
    run_audit([])


def cmd_backtest(args):
    from scripts.backtest_roi_analysis import main as run_backtest
    run_backtest()

# --- NEW COMMANDS ---

def cmd_list_models(args):
    """List models in registry."""
    registry = ModelRegistry()
    models = registry.list_models()
    
    print(f"\nModel Registry: {registry.model_name}")
    print("=" * 80)
    print(f"{'Version':<10} | {'Stage':<12} | {'AUC':<6} | {'Created At':<20} | {'Note'}")
    print("-" * 80)
    
    for model in models:
        stage_emoji = {
            'Production': '✅',
            'Staging': '🟡',
            'Experimental': '🔬',
            'Archived': '📦',
        }.get(model.stage, '?')
        
        note = model.notes if model.notes else ""
        print(f"{stage_emoji} {model.version:<8} | {model.stage:<12} | {model.auc:.3f}  | {model.trained_at[:19]:<20} | {note}")
    print("=" * 80)
    print()

def cmd_promote_model(args):
    """Promote model to new stage."""
    registry = ModelRegistry()
    registry.transition_stage(args.version, args.stage)
    print(f"✅ Promoted {args.version} to {args.stage}")

def cmd_set_serving_config(args):
    """Configure serving behavior."""
    config = ServingConfig(
        canary_percentage=args.canary,
        shadow_mode=args.shadow,
    )
    
    # Save config to file
    config_path = "config/serving.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"✅ Serving config updated:")
    print(f"   Canary Percentage: {args.canary*100}%")
    print(f"   Shadow Mode: {args.shadow}")


def main():
    parser = argparse.ArgumentParser(description="Tennis Prediction System")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Existing commands
    scrape = subparsers.add_parser("scrape")
    scrape.add_argument("mode", choices=["historical", "upcoming", "players"])
    scrape.add_argument("--top", type=int, default=50)
    scrape.add_argument("--pages", type=int, default=10)
    scrape.add_argument("--days", type=int, default=7)
    scrape.add_argument("--ids")
    scrape.add_argument("--resume", action="store_true")
    scrape.add_argument("--no-details", action="store_true")
    scrape.set_defaults(func=cmd_scrape)
    
    train = subparsers.add_parser("train")
    train.set_defaults(func=cmd_train)
    
    predict = subparsers.add_parser("predict")
    predict.add_argument("--days", type=int, default=7)
    predict.add_argument("--limit", type=int, default=10)
    predict.add_argument("--min-odds", type=float, default=1.5)
    predict.add_argument("--max-odds", type=float, default=3.0)
    predict.add_argument("--confidence", type=float, default=0.55)
    predict.add_argument("--no-scrape", action="store_true")
    predict.set_defaults(func=cmd_predict)
    
    audit = subparsers.add_parser("audit")
    audit.set_defaults(func=cmd_audit)
    
    backtest = subparsers.add_parser("backtest")
    backtest.set_defaults(func=cmd_backtest)
    
    # New Commands
    list_models = subparsers.add_parser('list-models', help='List model versions')
    list_models.set_defaults(func=cmd_list_models)
    
    promote = subparsers.add_parser('promote', help='Promote model version')
    promote.add_argument('version', help='Model version (e.g., v1.0.0)')
    promote.add_argument('stage', choices=['Staging', 'Production', 'Archived'])
    promote.set_defaults(func=cmd_promote_model)

    config_serving = subparsers.add_parser('serving-config', help='Configure serving settings')
    config_serving.add_argument('--canary', type=float, default=0.0, help='Canary percentage (0.0 - 1.0)')
    config_serving.add_argument('--shadow', action='store_true', help='Enable shadow mode')
    config_serving.set_defaults(func=cmd_set_serving_config)
    
    args = parser.parse_args()
    
    # Initialize correlation ID for this run
    correlation_id = str(uuid.uuid4())
    CORRELATION_ID.set(correlation_id)
    
    start_time = time.time()
    
    try:
        args.func(args)
    except Exception as e:
        logger.log_error("command_failed", error=str(e), exc_info=True)
        sys.exit(1)
    finally:
        duration = time.time() - start_time
        logger.log_event('command_completed', duration_seconds=duration)
        log_metrics_to_stdout()

if __name__ == "__main__":
    main()
