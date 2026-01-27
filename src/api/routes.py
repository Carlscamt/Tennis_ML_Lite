from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from datetime import datetime
import time

from src.pipeline import TennisPipeline
from src.api.schema import PredictionRequest, MatchPrediction, BatchPredictionResponse
from src.utils.observability import Logger, get_metrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

router = APIRouter()
logger = Logger(__name__)
metrics = get_metrics()

# Initialize pipeline (Singleton-ish for now, though instantiation per request is safer for thread-safety 
# if not designed to be shared, but pipeline loads models which is heavy. 
# TennisPipeline manages its own ModelServer state which should be thread-safe.)
pipeline = TennisPipeline()

@router.get("/health")
def health_check():
    """
    Service health check.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_version": "v1.0.0" # TODO: Fetch dynamic version
    }

@router.get("/metrics")
def metrics_endpoint():
    """
    Expose Prometheus metrics.
    """
    return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

@router.post("/predict", response_model=BatchPredictionResponse)
def generate_predictions(request: PredictionRequest):
    """
    Trigger a batch prediction run for upcoming matches.
    """
    logger.log_event("api_predict_request", days=request.days)
    
    try:
        # Run prediction pipeline
        # Note: This is blocking. For heavy loads, should be async or background task.
        # But for this use-case (batch/daily), blocking is acceptable for MVP.
        predictions_df = pipeline.predict_upcoming(
            days=request.days,
            min_confidence=request.min_confidence,
            scrape_unknown=True # Always scrape if needed
        )
        
        if len(predictions_df) == 0:
            return {
                "count": 0,
                "predictions": [],
                "generated_at": datetime.now().isoformat()
            }
            
        # Transform Polars DataFrame to List[MatchPrediction]
        # Using iter_rows(named=True)
        results = []
        for row in predictions_df.iter_rows(named=True):
            results.append(MatchPrediction(
                player_name=row['player_name'],
                opponent_name=row['opponent_name'],
                match_date=str(row['match_date']),
                tournament_name=row.get('tournament_name'),
                round_name=row.get('round_name'),
                surface=row.get('surface'),
                odds_player=row.get('odds_player'),
                odds_opponent=row.get('odds_opponent'),
                model_prob=row.get('model_prob', 0.0),
                edge=row.get('edge', 0.0),
                model_version=row.get('model_version', 'unknown'),
                serving_mode=row.get('serving_mode', 'unknown')
            ))
            
        return {
            "count": len(results),
            "predictions": results,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.log_error("api_predict_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
