"""
Prediction flow - Batch predictions and value bet detection.
"""
from prefect import flow, task
from prefect.logging import get_run_logger
import polars as pl
from pathlib import Path
from typing import List, Dict


@task(name="load-upcoming-matches")
def load_upcoming_task(days: int = 7) -> pl.DataFrame:
    """Load upcoming matches for prediction."""
    logger = get_run_logger()
    
    path = Path("data/upcoming.parquet")
    if not path.exists():
        logger.warning("No upcoming matches file found")
        return pl.DataFrame()
    
    df = pl.read_parquet(path)
    logger.info(f"Loaded {len(df)} upcoming matches")
    return df


@task(name="run-predictions")
def run_predictions_task(df: pl.DataFrame) -> pl.DataFrame:
    """Generate predictions for matches."""
    from src.model.predictor import Predictor
    from src.model.registry import ModelRegistry
    
    logger = get_run_logger()
    
    if df.is_empty():
        logger.warning("No matches to predict")
        return df
    
    # Get production model
    registry = ModelRegistry()
    try:
        version, model_path = registry.get_production_model()
    except RuntimeError:
        # Fallback to staging
        result = registry.get_challenger_model()
        if result is None:
            logger.error("No model available for predictions")
            return df
        version, model_path = result
    
    logger.info(f"Using model {version}")
    
    predictor = Predictor(model_path=Path(model_path))
    predictions = predictor.predict(df)
    
    logger.info(f"Generated {len(predictions)} predictions")
    return predictions


@task(name="filter-value-bets")
def filter_value_bets_task(df: pl.DataFrame, min_edge: float = 0.05) -> pl.DataFrame:
    """Filter predictions to value bets only."""
    logger = get_run_logger()
    
    if df.is_empty():
        logger.warning("No predictions to filter")
        return df
    
    # Filter for value bets
    if "edge" in df.columns:
        value_bets = df.filter(pl.col("edge") >= min_edge)
        logger.info(f"Found {len(value_bets)} value bets (edge >= {min_edge})")
        return value_bets
    
    logger.warning("No 'edge' column found, returning all predictions")
    return df


@task(name="save-predictions")
def save_predictions_task(df: pl.DataFrame, output_path: str) -> str:
    """Save predictions to file."""
    import json
    
    logger = get_run_logger()
    
    if df.is_empty():
        logger.warning("No predictions to save")
        return ""
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON for readability
    records = df.to_dicts()
    with open(output, "w") as f:
        json.dump(records, f, indent=2, default=str)
    
    logger.info(f"Saved {len(records)} predictions to {output}")
    return str(output)


@flow(name="batch-predictions", log_prints=True)
def batch_predictions_flow(
    days: int = 7,
    min_edge: float = 0.05,
    output_path: str = "results/value_bets.json"
) -> str:
    """
    Generate batch predictions and identify value bets.
    
    Args:
        days: Days of upcoming matches to predict
        min_edge: Minimum edge threshold for value bets
        output_path: Output file path
        
    Returns:
        Path to saved predictions file
    """
    logger = get_run_logger()
    logger.info(f"Starting batch predictions (days={days}, min_edge={min_edge})")
    
    # Load upcoming matches
    df = load_upcoming_task(days)
    
    if df.is_empty():
        logger.warning("No upcoming matches to predict")
        return ""
    
    # Generate predictions
    predictions = run_predictions_task(df)
    
    # Filter value bets
    value_bets = filter_value_bets_task(predictions, min_edge)
    
    # Save results
    result_path = save_predictions_task(value_bets, output_path)
    
    logger.info(f"Batch predictions complete: {len(value_bets)} value bets")
    return result_path
