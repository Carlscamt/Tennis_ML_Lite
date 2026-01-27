from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class PredictionRequest(BaseModel):
    """
    Request parameters for triggering a batch prediction run.
    """
    days: int = Field(default=7, ge=1, le=30, description="Number of days ahead to predict")
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum model confidence filter")
    force_refresh: bool = Field(default=False, description="Force fresh scrape even if cache exists")

class MatchPrediction(BaseModel):
    """
    Single match prediction result.
    """
    player_name: str
    opponent_name: str
    match_date: str
    tournament_name: Optional[str] = None
    round_name: Optional[str] = None
    surface: Optional[str] = None
    
    odds_player: Optional[float] = None
    odds_opponent: Optional[float] = None
    
    model_prob: float
    edge: float
    
    model_version: str
    serving_mode: str

class BatchPredictionResponse(BaseModel):
    """
    Response containing list of predictions.
    """
    count: int
    predictions: List[MatchPrediction]
    generated_at: str
