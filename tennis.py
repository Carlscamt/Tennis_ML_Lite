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
from src.serving.batch_job import run_batch_job, BatchOrchestrator

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
    
    # Filter value bets
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
    
    # Output to file if requested
    if args.output:
        out_path = Path(args.output)
        try:
            if out_path.suffix == '.csv':
                value_bets.write_csv(out_path)
            elif out_path.suffix == '.json':
                value_bets.write_json(out_path, row_oriented=True)
            elif out_path.suffix == '.parquet':
                value_bets.write_parquet(out_path)
            else:
                value_bets.write_csv(out_path)
            print(f"[OK] Saved to: {out_path}")
        except Exception as e:
            logger.log_error("save_failed", error=str(e))
            print(f"Error saving: {e}")
    
    # ASCII formatted output
    _print_predictions_ascii(value_bets, args.days)


def _print_predictions_ascii(value_bets, days):
    """Print predictions as a compact ASCII table."""
    
    print(f"\n{'':=^90}")
    print(f"VALUE BETS - Next {days} Days ({len(value_bets)} found)".center(90))
    print(f"{'':=^90}")
    
    # Table header
    print(f"| {'BET ON':<20} | {'vs':<18} | {'Prob':>5} | {'Odds':>5} | {'Edge':>6} | {'Tournament':<18} |")
    print(f"|{'-'*22}|{'-'*20}|{'-'*7}|{'-'*7}|{'-'*8}|{'-'*20}|")
    
    # Group by tournament
    tournaments = {}
    for row in value_bets.iter_rows(named=True):
        tourney = row.get('tournament_name', 'Unknown')
        if tourney not in tournaments:
            tournaments[tourney] = []
        tournaments[tourney].append(row)
    
    # Print rows grouped by tournament
    for tourney, matches in tournaments.items():
        for row in matches:
            player = row['player_name'][:20]
            opponent = row['opponent_name'][:18]
            prob = row.get('model_prob', 0) * 100
            odds = row.get('odds_player', 0)
            edge = row.get('edge', 0) * 100
            tourney_short = tourney[:18]
            
            # Edge indicator
            if edge >= 15:
                edge_str = f"+{edge:4.1f}%!"
            elif edge >= 10:
                edge_str = f"+{edge:4.1f}%"
            else:
                edge_str = f"+{edge:4.1f}%"
            
            print(f"| {player:<20} | {opponent:<18} | {prob:4.0f}% | {odds:5.2f} | {edge_str:>6} | {tourney_short:<18} |")
    
    print(f"|{'='*22}|{'='*20}|{'='*7}|{'='*7}|{'='*8}|{'='*20}|")
    print(f" Filter: Edge > 5% | Confidence > 55% | Odds 1.50-3.00\n")


def cmd_audit(args):
    from scripts.model_audit import main as run_audit
    run_audit([])


def cmd_backtest(args):
    from scripts.backtest_roi_analysis import main as run_backtest
    run_backtest()

# --- MODEL REGISTRY COMMANDS ---

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
            'Production': '[PROD]',
            'Staging': '[STAGE]',
            'Experimental': '[EXP]',
            'Archived': '[ARCH]',
        }.get(model.stage, '?')
        
        note = model.notes if model.notes else ""
        print(f"{stage_emoji} {model.version:<8} | {model.stage:<12} | {model.auc:.3f}  | {model.trained_at[:19]:<20} | {note}")
    print("=" * 80)
    print()

def cmd_promote_model(args):
    """Promote model to new stage."""
    registry = ModelRegistry()
    registry.transition_stage(args.version, args.stage)
    print(f"Promoted {args.version} to {args.stage}")

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
    
    print(f"Serving config updated:")
    print(f"   Canary Percentage: {args.canary*100}%")
    print(f"   Shadow Mode: {args.shadow}")

# --- BATCH SERVING COMMANDS ---

def cmd_batch_run(args):
    """Trigger daily batch job (Scheduler)."""
    print(f"Triggering Batch Job (Fetch {args.days} days)...")
    status = run_batch_job(force=args.force, days=args.days)
    print(f"Batch Job Finished: {status}")


def cmd_run_flow(args):
    """Execute a Prefect flow."""
    from src.flows import (
        scrape_historical_flow,
        scrape_upcoming_flow,
        build_features_flow,
        train_model_flow,
        batch_predictions_flow,
        daily_pipeline_flow,
        full_retrain_flow,
    )
    
    flows = {
        "scrape-historical": lambda: scrape_historical_flow(
            top_players=args.top,
            pages=args.pages
        ),
        "scrape-upcoming": lambda: scrape_upcoming_flow(days=args.days),
        "build-features": build_features_flow,
        "train-model": train_model_flow,
        "batch-predictions": lambda: batch_predictions_flow(
            days=args.days,
            min_edge=args.min_edge
        ),
        "daily-pipeline": lambda: daily_pipeline_flow(
            scrape_days=args.days,
            min_edge=args.min_edge,
            skip_scrape=args.skip_scrape,
            skip_features=args.skip_features
        ),
        "full-retrain": lambda: full_retrain_flow(
            top_players=args.top,
            pages=args.pages
        ),
    }
    
    if args.list:
        print("\nAvailable flows:")
        for name in flows.keys():
            print(f"  - {name}")
        return
    
    flow_func = flows.get(args.flow_name)
    if not flow_func:
        print(f"ERROR: Unknown flow '{args.flow_name}'")
        print(f"Available: {', '.join(flows.keys())}")
        return
    
    print(f"\nExecuting flow: {args.flow_name}")
    print("=" * 50)
    
    result = flow_func()
    
    print("=" * 50)
    print(f"Flow completed. Result: {result}")


def cmd_show_predictions(args):
    """Instant serving command (Reads Cache)."""
    orch = BatchOrchestrator()
    cached = orch.get_predictions()
    
    if cached.is_empty():
        print("WARN: No cached predictions found. Run 'batch-run' first.")
        return

    # Filter and display (same logic as predict but instant)
    print(f"\nINSTANT SERVING (Cache Count: {len(cached)})")
    
    value_bets = cached.filter(
        (cached["edge"] > 0.05) &
        (cached["odds_player"] >= args.min_odds)
    ).sort("edge", descending=True).head(args.limit)
    
    if len(value_bets) == 0:
        print("No top value bets found in cache.")
        return
        
    for i, row in enumerate(value_bets.iter_rows(named=True), 1):
        player = row['player_name']
        opponent = row['opponent_name']
        prob = row.get('model_prob', 0) * 100
        odds = row.get('odds_player', 0)
        edge = row.get('edge', 0) * 100
        match_date = row.get('match_date', 'Unknown')
        
        print(f"#{i} {player} vs {opponent} | {match_date} | Edge: {edge:.1f}%")


def cmd_showdown(args):
    """Run tournament showdown simulation."""
    from src.showdown import TournamentSimulator, BracketVisualizer
    from config import DATA_DIR
    
    logger.log_event('showdown_command_started', tournament=args.tournament, year=args.year)
    
    # Initialize simulator
    data_path = DATA_DIR / "tennis.parquet"
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print("Run 'python tennis.py scrape historical' first.")
        return 1
    
    simulator = TournamentSimulator(data_path)
    
    # List tournaments mode
    if args.list:
        print(f"\n=== Available Tournaments (Year: {args.year or 'All'}) ===")
        tournaments = simulator.list_available_tournaments(year=args.year)
        for t in tournaments[:20]:  # Show top 20
            print(f"  - {t['name']} ({t['year']}) - {t['matches']} matches")
        return
    
    # Validate required args for simulation
    if not args.tournament:
        print("ERROR: --tournament required. Use --list to see available tournaments.")
        return 1
    if not args.year:
        print("ERROR: --year required.")
        return 1
    
    try:
        print(f"\n[SHOWDOWN] {args.tournament} {args.year}")
        print("=" * 60)
        
        # Run showdown
        bracket, stats = simulator.run_showdown(args.tournament, args.year)
        
        # ASCII terminal bracket
        if args.ascii:
            _print_ascii_bracket(bracket, stats)
        else:
            # Summary stats
            print(f"\n[RESULTS]")
            print(f"   Total: {stats.total_matches} | Predicted: {stats.predicted_matches} | Correct: {stats.correct_predictions}")
            print(f"   Accuracy: {stats.accuracy * 100:.1f}% | Confidence: {stats.avg_confidence * 100:.1f}%")
            
            if stats.upsets_total > 0:
                print(f"   Upsets: {stats.upsets_predicted}/{stats.upsets_total}")
            
            print(f"\n[ACCURACY BY ROUND]")
            for round_name, acc in stats.accuracy_by_round.items():
                bar = "#" * int(acc * 20) + "-" * (20 - int(acc * 20))
                print(f"   {round_name:20s} [{bar}] {acc * 100:.0f}%")
            
            # Generate HTML
            visualizer = BracketVisualizer()
            output_path = Path(args.output) if args.output else None
            html_path = visualizer.render_html(bracket, stats, output_path)
            print(f"\n[OK] Saved to: {html_path}")
            
            if args.json:
                json_path = visualizer.export_json(bracket, stats)
                print(f"     JSON: {json_path}")
        
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Use --list to see available tournaments.")
        return 1
    except Exception as e:
        logger.log_error("showdown_failed", error=str(e))
        print(f"ERROR: Showdown failed - {e}")
        return 1


def _print_ascii_bracket(bracket, stats):
    """Print ASCII tournament bracket to terminal."""
    
    # Get last 4 rounds
    all_rounds = sorted(bracket.rounds.keys())
    display_rounds = all_rounds[-4:] if len(all_rounds) > 4 else all_rounds
    
    # Header
    print(f"\n{'':=^80}")
    print(f"{bracket.config.name.upper()} {bracket.config.year}".center(80))
    print(f"Accuracy: {stats.accuracy*100:.1f}% ({stats.correct_predictions}/{stats.predicted_matches})".center(80))
    print(f"{'':=^80}\n")
    
    # Print each round
    for round_num in display_rounds:
        matches = bracket.rounds[round_num]
        if not matches:
            continue
        
        round_name = matches[0].round_name
        print(f"--- {round_name} ---")
        
        for match in matches:
            _print_match_line(match)
        
        print()
    
    # Footer
    print(f"{'-'*80}")
    print(f" Legend: [+] Correct  [-] Wrong  * = Winner  << = Model Pick")
    print(f" Upsets: {stats.upsets_predicted}/{stats.upsets_total} caught | Confidence: {stats.avg_confidence*100:.0f}%")
    print(f"{'-'*80}\n")


def _print_match_line(match):
    """Print a single match on one line."""
    # Status indicator
    if match.prediction_correct is True:
        status = "[+]"
    elif match.prediction_correct is False:
        status = "[-]"
    else:
        status = "[?]"
    
    # Player names
    p1 = match.player1_name[:18] if match.player1_name else "TBD"
    p2 = match.player2_name[:18] if match.player2_name else "TBD"
    
    # Winner marker
    p1_mark = "*" if match.actual_winner_id == match.player1_id else " "
    p2_mark = "*" if match.actual_winner_id == match.player2_id else " "
    
    # Model pick
    p1_pick = "<<" if match.model_winner_id == match.player1_id else "  "
    p2_pick = "<<" if match.model_winner_id == match.player2_id else "  "
    
    # Confidence
    conf = f"{match.model_confidence*100:.0f}%" if match.model_confidence else "  "
    
    print(f"  {status} {p1_mark}{p1:<18}{p1_pick} vs {p2_mark}{p2:<18}{p2_pick} ({conf})")


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
    predict.add_argument("--limit", type=int, default=50)
    predict.add_argument("--min-odds", type=float, default=1.5)
    predict.add_argument("--max-odds", type=float, default=3.0)
    predict.add_argument("--confidence", type=float, default=0.55)
    predict.add_argument("--no-scrape", action="store_true")
    predict.add_argument("--output", help="Save predictions to file (supports .csv, .json, .parquet)")
    predict.set_defaults(func=cmd_predict)
    
    audit = subparsers.add_parser("audit")
    audit.set_defaults(func=cmd_audit)
    
    backtest = subparsers.add_parser("backtest")
    backtest.set_defaults(func=cmd_backtest)
    
    # Registry Commands
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

    # Batch Serving Commands
    batch = subparsers.add_parser('batch-run', help='Execute daily batch job')
    batch.add_argument('--days', type=int, default=7)
    batch.add_argument('--force', action='store_true', help='Force run even if cache valid')
    batch.set_defaults(func=cmd_batch_run)

    show = subparsers.add_parser('show-predictions', help='Instant cached predictions')
    show.add_argument("--limit", type=int, default=10)
    show.add_argument("--min-odds", type=float, default=1.5)
    show.set_defaults(func=cmd_show_predictions)
    
    # Showdown Command
    showdown = subparsers.add_parser('showdown', help='Tournament bracket simulation')
    showdown.add_argument('--tournament', '-t', type=str, help='Tournament name (e.g., "Australian Open")')
    showdown.add_argument('--year', '-y', type=int, help='Tournament year')
    showdown.add_argument('--output', '-o', type=str, help='Output HTML path')
    showdown.add_argument('--list', '-l', action='store_true', help='List available tournaments')
    showdown.add_argument('--json', action='store_true', help='Also export JSON data')
    showdown.add_argument('--ascii', '-a', action='store_true', help='Print ASCII bracket to terminal')
    showdown.set_defaults(func=cmd_showdown)
    
    # Prefect Flow Commands
    run_flow = subparsers.add_parser('run-flow', help='Execute a Prefect flow')
    run_flow.add_argument('flow_name', nargs='?', help='Flow name (e.g., daily-pipeline)')
    run_flow.add_argument('--list', '-l', action='store_true', help='List available flows')
    run_flow.add_argument('--top', type=int, default=50, help='Top players for scraping')
    run_flow.add_argument('--pages', type=int, default=10, help='Pages per player')
    run_flow.add_argument('--days', type=int, default=7, help='Days for upcoming/predictions')
    run_flow.add_argument('--min-edge', type=float, default=0.05, help='Minimum edge threshold')
    run_flow.add_argument('--skip-scrape', action='store_true', help='Skip scraping step')
    run_flow.add_argument('--skip-features', action='store_true', help='Skip feature building')
    run_flow.set_defaults(func=cmd_run_flow)
    
    args = parser.parse_args()
    
    # Initialize correlation ID for this run
    correlation_id = str(uuid.uuid4())
    CORRELATION_ID.set(correlation_id)
    
    start_time = time.time()
    
    try:
        args.func(args)
    except Exception as e:
        try:
            logger.log_error("command_failed", error=str(e), exc_info=True)
        except UnicodeEncodeError:
            # Fallback for Windows encoding issues
            print(f"ERROR: Command failed with {type(e).__name__}. (Details omitted due to encoding error)")
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        duration = time.time() - start_time
        logger.log_event('command_completed', duration_seconds=duration)
        log_metrics_to_stdout()

if __name__ == "__main__":
    main()
