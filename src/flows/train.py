"""
Training flow - Model training and registration.
"""
from prefect import flow, task
from prefect.logging import get_run_logger
import polars as pl
from pathlib import Path
from typing import Tuple, Dict, Any


@task(name="load-features")
def load_features_task(features_path: str = "data/processed/features_dataset.parquet") -> pl.DataFrame:
    """Load processed features for training."""
    logger = get_run_logger()
    
    path = Path(features_path)
    if not path.exists():
        logger.error(f"Features file not found: {path}")
        return pl.DataFrame()
    
    df = pl.read_parquet(path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df


@task(name="train-model")
def train_model_task(df: pl.DataFrame) -> Tuple[Any, Dict[str, float]]:
    """Train XGBoost model."""
    from src.model.trainer import ModelTrainer
    
    logger = get_run_logger()
    
    if df.is_empty():
        logger.error("Empty dataframe, cannot train")
        return None, {}
    
    trainer = ModelTrainer()
    model, metrics = trainer.train(df)
    
    logger.info(f"Training complete - AUC: {metrics.get('auc', 'N/A'):.4f}")
    return model, metrics


@task(name="register-model")
def register_model_task(model: Any, metrics: Dict[str, float]) -> str:
    """Register trained model in registry."""
    from src.model.registry import ModelRegistry
    
    logger = get_run_logger()
    
    if model is None:
        logger.error("No model to register")
        return ""
    
    registry = ModelRegistry()
    version = registry.register(model, metrics, stage="Experimental")
    
    logger.info(f"Registered model as {version}")
    return version


@flow(name="train-model", log_prints=True)
def train_model_flow(features_path: str = "data/processed/features_dataset.parquet") -> str:
    """
    Train a new model on processed features.
    
    Args:
        features_path: Path to features parquet file
        
    Returns:
        Registered model version string
    """
    logger = get_run_logger()
    logger.info("Starting model training pipeline")
    
    # Load features
    df = load_features_task(features_path)
    
    if df.is_empty():
        logger.error("No features available for training")
        return ""
    
    # Train model
    model, metrics = train_model_task(df)
    
    if model is None:
        logger.error("Training failed")
        return ""
    
    # Register
    version = register_model_task(model, metrics)
    
    logger.info(f"Training pipeline complete: {version}")
    return version
