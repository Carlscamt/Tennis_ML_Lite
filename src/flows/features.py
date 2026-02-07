"""
Features flow - ETL for feature engineering.
"""
from prefect import flow, task
from prefect.logging import get_run_logger
import polars as pl
from pathlib import Path


@task(name="load-raw-data")
def load_raw_data_task() -> pl.DataFrame:
    """Load and merge raw parquet files."""
    from src.schema import merge_datasets
    
    logger = get_run_logger()
    
    raw_dir = Path("data/raw")
    files = list(raw_dir.glob("*.parquet"))
    
    if not files:
        logger.warning("No raw parquet files found")
        return pl.DataFrame()
    
    dfs = [pl.read_parquet(f) for f in files]
    merged = merge_datasets(dfs) if len(dfs) > 1 else dfs[0]
    
    logger.info(f"Loaded {len(merged)} records from {len(files)} files")
    return merged


@task(name="engineer-features")
def engineer_features_task(df: pl.DataFrame) -> pl.DataFrame:
    """Apply feature engineering transformations."""
    from src.transform.features import FeatureEngineer
    
    logger = get_run_logger()
    
    if df.is_empty():
        logger.warning("Empty dataframe, skipping feature engineering")
        return df
    
    engineer = FeatureEngineer()
    features_df = engineer.transform(df)
    
    logger.info(f"Generated {len(features_df.columns)} features for {len(features_df)} records")
    return features_df


@task(name="validate-schema")
def validate_schema_task(df: pl.DataFrame) -> pl.DataFrame:
    """Validate dataframe against schema."""
    from src.schema import SchemaValidator
    
    logger = get_run_logger()
    
    if df.is_empty():
        logger.warning("Empty dataframe, skipping validation")
        return df
    
    validator = SchemaValidator()
    validated = validator.validate(df)
    
    logger.info("Schema validation passed")
    return validated


@task(name="save-features")
def save_features_task(df: pl.DataFrame, output_path: str) -> str:
    """Save features to parquet."""
    logger = get_run_logger()
    
    if df.is_empty():
        logger.warning("Empty dataframe, not saving")
        return ""
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output, compression="snappy")
    
    logger.info(f"Saved {len(df)} records to {output}")
    return str(output)


@flow(name="build-features", log_prints=True)
def build_features_flow() -> str:
    """
    Build features from raw data.
    
    Loads raw parquet files, applies feature engineering,
    validates schema, and saves processed features.
    
    Returns:
        Path to saved features parquet file
    """
    logger = get_run_logger()
    logger.info("Starting feature build pipeline")
    
    # Load raw data
    raw_df = load_raw_data_task()
    
    if raw_df.is_empty():
        logger.error("No raw data available")
        return ""
    
    # Engineer features
    features_df = engineer_features_task(raw_df)
    
    # Validate schema
    validated_df = validate_schema_task(features_df)
    
    # Save
    output_path = "data/processed/features_dataset.parquet"
    result_path = save_features_task(validated_df, output_path)
    
    logger.info(f"Feature build complete: {len(validated_df)} records")
    return result_path
