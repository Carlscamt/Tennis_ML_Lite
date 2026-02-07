"""
Daily Pipeline flow - Orchestrates the full daily workflow.
"""
from prefect import flow
from prefect.logging import get_run_logger
from typing import Dict, Any


@flow(name="daily-pipeline", log_prints=True)
def daily_pipeline_flow(
    scrape_days: int = 7,
    min_edge: float = 0.05,
    skip_scrape: bool = False,
    skip_features: bool = False
) -> Dict[str, Any]:
    """
    Orchestrate the complete daily prediction pipeline.
    
    Runs in sequence:
    1. Scrape upcoming matches
    2. Build/update features
    3. Generate predictions
    4. Output value bets
    
    Args:
        scrape_days: Days of upcoming matches to fetch
        min_edge: Minimum edge for value bet filter
        skip_scrape: Skip scraping step (use cached data)
        skip_features: Skip feature building (use existing features)
        
    Returns:
        Summary dict with paths and counts
    """
    from .scrape import scrape_upcoming_flow
    from .features import build_features_flow
    from .predict import batch_predictions_flow
    
    logger = get_run_logger()
    logger.info("=" * 50)
    logger.info("DAILY PIPELINE STARTED")
    logger.info("=" * 50)
    
    results = {
        "scrape_path": None,
        "features_path": None,
        "predictions_path": None,
        "status": "started"
    }
    
    # Step 1: Scrape upcoming matches
    if not skip_scrape:
        logger.info("Step 1/3: Scraping upcoming matches")
        results["scrape_path"] = scrape_upcoming_flow(days=scrape_days)
    else:
        logger.info("Step 1/3: Skipped scraping (using cached data)")
    
    # Step 2: Build features
    if not skip_features:
        logger.info("Step 2/3: Building features")
        results["features_path"] = build_features_flow()
    else:
        logger.info("Step 2/3: Skipped features (using existing)")
    
    # Step 3: Run predictions
    logger.info("Step 3/3: Generating predictions")
    results["predictions_path"] = batch_predictions_flow(
        days=scrape_days,
        min_edge=min_edge
    )
    
    results["status"] = "completed"
    
    logger.info("=" * 50)
    logger.info("DAILY PIPELINE COMPLETED")
    logger.info(f"Predictions: {results['predictions_path']}")
    logger.info("=" * 50)
    
    return results


@flow(name="full-retrain-pipeline", log_prints=True)
def full_retrain_flow(
    top_players: int = 50,
    pages: int = 10,
    ranking_type: str = "atp_singles"
) -> Dict[str, Any]:
    """
    Full pipeline including historical scraping and retraining.
    
    Use this for periodic model updates (weekly/monthly).
    
    Runs:
    1. Scrape historical data
    2. Build features
    3. Train new model
    4. Run predictions
    """
    from .scrape import scrape_historical_flow
    from .features import build_features_flow
    from .train import train_model_flow
    from .predict import batch_predictions_flow
    
    logger = get_run_logger()
    logger.info("=" * 50)
    logger.info("FULL RETRAIN PIPELINE STARTED")
    logger.info("=" * 50)
    
    results = {}
    
    # Historical scrape
    logger.info("Step 1/4: Scraping historical data")
    results["historical_path"] = scrape_historical_flow(
        top_players=top_players,
        pages=pages,
        ranking_type=ranking_type
    )
    
    # Build features
    logger.info("Step 2/4: Building features")
    results["features_path"] = build_features_flow()
    
    # Train model
    logger.info("Step 3/4: Training model")
    results["model_version"] = train_model_flow()
    
    # Generate predictions
    logger.info("Step 4/4: Generating predictions")
    results["predictions_path"] = batch_predictions_flow()
    
    results["status"] = "completed"
    
    logger.info("=" * 50)
    logger.info("FULL RETRAIN PIPELINE COMPLETED")
    logger.info(f"New model: {results['model_version']}")
    logger.info("=" * 50)
    
    return results
