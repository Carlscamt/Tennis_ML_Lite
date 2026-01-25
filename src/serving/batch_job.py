"""
Batch Orchestrator Module.
Manages daily scrape -> predict -> cache workflow.
"""
import sys
import time
from pathlib import Path
from datetime import datetime, date
import polars as pl

# Use absolute imports relative to project root
from src.pipeline import TennisPipeline
from src.serving.cache import PredictionCache
from src.utils.observability import Logger, get_metrics

logger = Logger(__name__)

class BatchOrchestrator:
    """
    Manages daily batch jobs.
    """
    
    def __init__(self):
        self.pipeline = TennisPipeline()
        self.cache = PredictionCache()
        
    def run_daily_batch(self, force: bool = False, days: int = 7) -> str:
        """
        Execute daily batch: Scrape -> Predict -> Cache.
        Returns status message.
        """
        # 1. Check Cache Freshman
        if not force:
            cached = self.cache.get(max_age_hours=24) # Daily cycle
            if cached is not None:
                logger.log_event("batch_job_skipped_cache_fresh")
                print(f"✅ Cache is fresh (Updated: {datetime.now()}). Use --force to re-run.")
                return "SKIPPED"

        logger.log_event("batch_job_started")
        start_time = time.time()
        
        try:
            # 2. Scrape & Predict (Pipeline encapsulates this logic under predict_upcoming)
            # predict_upcoming handles scraping internally if cache stale or missing
            # It also handles feature engineering and model inference
            predictions = self.pipeline.predict_upcoming(
                days=days,
                min_odds=1.1, # Wide range for cache
                max_odds=10.0,
                min_confidence=0.0, # Cache everything, filter at serve time
                scrape_unknown=True 
            )
            
            if len(predictions) == 0:
                logger.log_error("batch_job_no_predictions")
                return "FAILED_NO_DATA"
                
            # 3. Cache Results
            self.cache.save(predictions, ttl_hours=24)
            
            duration = time.time() - start_time
            logger.log_event("batch_job_completed", duration_seconds=duration, count=len(predictions))
            return "SUCCESS"
            
        except Exception as e:
            logger.log_error("batch_job_failed", error=str(e), exc_info=True)
            return f"FAILED: {str(e)}"
            
    def get_predictions(self, max_age_hours: int = 24) -> pl.DataFrame:
        """Read directly from cache (Serving Layer)."""
        cached = self.cache.get(max_age_hours=max_age_hours)
        if cached is not None:
            return cached
            
        # Fallback? If no cache, should we run ad-hoc prediction?
        # User requested "Batch serving is optimal". "Serving layer - Fast lookups".
        # If cache miss, we can either trigger batch or return empty.
        # Triggering batch might be slow (scraping). 
        # Let's return empty and log warning, prompting user to run batch.
        logger.log_event("serving_cache_miss")
        return pl.DataFrame()

def run_batch_job(force=False, days=7):
    orch = BatchOrchestrator()
    orch.run_daily_batch(force=force, days=days)
