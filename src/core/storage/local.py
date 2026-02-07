"""
Local Parquet Repository - Default storage implementation.

Stores data as Parquet files on the local filesystem.
"""
import polars as pl
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class LocalParquetRepository:
    """
    Local filesystem storage using Parquet format.
    
    Directory structure:
        {base_path}/
        ├── raw/                 # Raw scraped data
        │   ├── {partition}.parquet
        ├── processed/           # Processed features
        │   ├── {name}.parquet
        └── upcoming/            # Upcoming matches
            └── upcoming.parquet
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        
        # Ensure directories exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
    
    def load_raw(self, partition: str) -> pl.DataFrame:
        """Load raw data from a partition."""
        path = self.raw_path / f"{partition}.parquet"
        
        if not path.exists():
            logger.warning(f"Partition not found: {partition}")
            return pl.DataFrame()
        
        df = pl.read_parquet(path)
        logger.debug(f"Loaded {len(df)} rows from {partition}")
        return df
    
    def save_raw(self, df: pl.DataFrame, partition: str) -> None:
        """Save raw data to a partition."""
        if df.is_empty():
            logger.warning(f"Empty dataframe, not saving partition: {partition}")
            return
        
        path = self.raw_path / f"{partition}.parquet"
        df.write_parquet(path, compression="snappy")
        logger.info(f"Saved {len(df)} rows to {partition}")
    
    def load_features(self, name: str = "features_dataset") -> pl.DataFrame:
        """Load processed features."""
        path = self.processed_path / f"{name}.parquet"
        
        if not path.exists():
            logger.warning(f"Features not found: {name}")
            return pl.DataFrame()
        
        df = pl.read_parquet(path)
        logger.debug(f"Loaded {len(df)} feature rows from {name}")
        return df
    
    def save_features(self, df: pl.DataFrame, name: str = "features_dataset") -> None:
        """Save processed features."""
        if df.is_empty():
            logger.warning(f"Empty dataframe, not saving features: {name}")
            return
        
        path = self.processed_path / f"{name}.parquet"
        df.write_parquet(path, compression="snappy")
        logger.info(f"Saved {len(df)} feature rows to {name}")
    
    def list_partitions(self, prefix: str = "") -> List[str]:
        """List available raw data partitions."""
        partitions = []
        
        for path in self.raw_path.glob("*.parquet"):
            name = path.stem
            if prefix and not name.startswith(prefix):
                continue
            partitions.append(name)
        
        return sorted(partitions)
    
    def exists(self, partition: str) -> bool:
        """Check if a partition exists."""
        path = self.raw_path / f"{partition}.parquet"
        return path.exists()
    
    def load_all_raw(self, prefix: str = "") -> pl.DataFrame:
        """Load and concatenate all raw partitions matching prefix."""
        partitions = self.list_partitions(prefix)
        
        if not partitions:
            logger.warning(f"No partitions found with prefix: {prefix}")
            return pl.DataFrame()
        
        dfs = [self.load_raw(p) for p in partitions]
        dfs = [df for df in dfs if not df.is_empty()]
        
        if not dfs:
            return pl.DataFrame()
        
        combined = pl.concat(dfs)
        logger.info(f"Loaded {len(combined)} total rows from {len(dfs)} partitions")
        return combined
