"""
Prediction Cache Module.
Handles storage and retrieval of predictions with TTL.
"""
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import polars as pl
import structlog
from src.utils.observability import Logger

logger = Logger(__name__)

class PredictionCache:
    """
    File-based prediction cache with Time-To-Live (TTL).
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "daily_predictions.parquet"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
    def save(self, predictions: pl.DataFrame, ttl_hours: int = 24) -> None:
        """
        Save predictions to cache.
        """
        try:
            # Write parquet
            predictions.write_parquet(self.cache_file)
            
            # Write metadata
            metadata = {
                "updated_at": datetime.now().isoformat(),
                "ttl_hours": ttl_hours,
                "count": len(predictions),
                "model_versions": predictions["model_version"].unique().to_list() if "model_version" in predictions.columns else []
            }
            
            with open(self.metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.log_event("cache_save_success", count=len(predictions), ttl=ttl_hours)
            
        except Exception as e:
            logger.log_error("cache_save_failed", error=str(e))
            raise

    def get(self, max_age_hours: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        Get cached predictions if valid.
        
        Args:
            max_age_hours: Override stored TTL if provided.
        """
        if not self.cache_file.exists() or not self.metadata_file.exists():
            return None
            
        try:
            with open(self.metadata_file, "r") as f:
                metadata = json.load(f)
                
            updated_at = datetime.fromisoformat(metadata["updated_at"])
            age_hours = (datetime.now() - updated_at).total_seconds() / 3600
            
            ttl = max_age_hours if max_age_hours is not None else metadata.get("ttl_hours", 24)
            
            if age_hours > ttl:
                logger.log_event("cache_stale_ignored", age_hours=age_hours, ttl=ttl)
                return None
                
            df = pl.read_parquet(self.cache_file)
            logger.log_event("cache_hit", age_hours=age_hours)
            return df
            
        except Exception as e:
            logger.log_error("cache_read_failed", error=str(e))
            return None

    def invalidate(self):
        """Clear cache."""
        try:
            if self.cache_file.exists():
                os.remove(self.cache_file)
            if self.metadata_file.exists():
                os.remove(self.metadata_file)
            logger.log_event("cache_invalidated")
        except Exception as e:
             logger.log_error("cache_invalidation_failed", error=str(e))
