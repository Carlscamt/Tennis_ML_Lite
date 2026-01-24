
"""
Advanced Model Serving (Canary, Shadow, Fallback).
"""
import asyncio
import random
import time
import os
import json
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass, asdict
import xgboost as xgb
import structlog
from src.utils.observability import get_metrics, Logger, CORRELATION_ID
from src.model.registry import ModelRegistry

logger = Logger(__name__)
metrics = get_metrics()

class ServingMode(str, Enum):
    CHAMPION_ONLY = "champion_only"
    CANARY = "canary"
    SHADOW = "shadow"
    FALLBACK = "fallback"

@dataclass
class PredictionResult:
    """Structured prediction result."""
    predictions: Union[np.ndarray, List]
    model_version: str
    latency_ms: float
    confidence_scores: Optional[Union[np.ndarray, List]] = None

@dataclass
class ServingConfig:
    """Model serving configuration."""
    canary_percentage: float = 0.0  # 0-1.0 (e.g., 0.1 = 10% to challenger)
    shadow_mode: bool = False      # Run challenger in parallel
    enable_fallback: bool = True   # Use challenger if champion fails
    
    @classmethod
    def from_env(cls) -> 'ServingConfig':
        """Load config from environment variables or config file."""
        # Try loading from local config json first
        config_path = "config/serving.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                    return cls(**data)
            except Exception as e:
                logger.log_error("failed_load_serving_config", error=str(e))
        
        return cls(
            canary_percentage=float(os.getenv('CANARY_PERCENTAGE', 0.0)),
            shadow_mode=os.getenv('SHADOW_MODE', 'false').lower() == 'true',
            enable_fallback=os.getenv('ENABLE_FALLBACK', 'true').lower() == 'true',
        )

class ModelServer:
    """
    Production model serving with canary, shadow, and fallback support.
    """
    
    def __init__(self, registry: ModelRegistry, config: ServingConfig):
        self.registry = registry
        self.config = config
        self.champion_model = None
        self.challenger_model = None
        self._load_models()
    
    def _load_xgboost_model(self, path: str) -> xgb.XGBClassifier:
        model = xgb.XGBClassifier()
        model.load_model(path)
        return model

    def _load_models(self):
        """Load champion (Production) and challenger (Staging) models."""
        try:
            # Load champion (always required)
            champion_version, champion_path = self.registry.get_production_model()
            self.champion_model = self._load_xgboost_model(champion_path)
            # Monkey-patch version for tracking
            self.champion_model.version = champion_version
            logger.log_event('champion_model_loaded', version=champion_version)
            
            # Load challenger (optional)
            challenger = self.registry.get_challenger_model()
            if challenger:
                challenger_version, challenger_path = challenger
                self.challenger_model = self._load_xgboost_model(challenger_path)
                self.challenger_model.version = challenger_version
                logger.log_event('challenger_model_loaded', version=challenger_version)
            else:
                logger.log_event('no_challenger_model')
                
        except Exception as e:
            logger.log_error('model_loading_failed', error=str(e), exc_info=True)
            # Don't crash init if only challenger fails? User wants robustness.
            # But if Champion fails, we can't serve.
            # raise RuntimeError(f"Failed to load models: {e}")
            # If no production model found (e.g. fresh install), we might want to allow empty init?
            pass
    
    async def predict_batch(
        self, 
        features: List[Dict[str, Any]], 
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Make prediction using production serving patterns (Async Wrapper).
        
        Since XGBoost is CPU bound and synchronous, async here is mostly simulating
        concurrent request handling in web server context (FastAPI).
        For CLI, it effectively runs sequentially unless we use ThreadPoolExecutor.
        """
        import asyncio
        loop = asyncio.get_running_loop()
        
        # Run blocking prediction in executor
        return await loop.run_in_executor(None, self._predict_sync, features, request_id)

    def _predict_sync(self, features: List[Dict[str, Any]], request_id: Optional[str]) -> Dict[str, Any]:
        """Synchronous implementation of serving logic."""
        if not self.champion_model:
            raise RuntimeError("Model Server not initialized with Production model")

        corr_id = CORRELATION_ID.get() or request_id or str(random.randint(100000, 999999))
        CORRELATION_ID.set(corr_id)
        
        start_time = time.time()
        
        # Prepare Features (Dict List -> Numpy)
        # Assuming features are homogeneous
        # Need to ensure correct column order matching model.
        # Ideally we pass feature names or dataframe to predict?
        # XGBoost expects DMatrix or numpy array. DataFrame preserves order if passed.
        # Here we receive List of Dicts.
        
        # Quick hack: get keys from first item.
        # In robust system, ModelRegistry should persist feature schema order!
        if not features:
            return {'predictions': []}
            
        feature_keys = list(features[0].keys())
        # Ideally sort optional? Or just convert values.
        feature_array = np.array([[row[k] for k in feature_keys] for row in features])
        
        champion_result = None
        challenger_result = None
        serving_mode = ServingMode.CHAMPION_ONLY
        
        try:
            # 1. Champion
            champion_start = time.time()
            champion_result = self._predict_single(self.champion_model, feature_array)
            champion_latency = (time.time() - champion_start) * 1000
            
            # 2. Challenger
            challenger_latency = 0
            if self.challenger_model:
                challenger_start = time.time()
                
                if self.config.shadow_mode:
                    # Shadow: Run and log, discard result
                    serving_mode = ServingMode.SHADOW
                    challenger_result = self._predict_single(self.challenger_model, feature_array)
                    # Log diffs?
                    self._log_shadow_diff(champion_result, challenger_result, corr_id)
                    challenger_result = None # Don't return it
                    challenger_latency = (time.time() - challenger_start) * 1000
                
                elif self.config.canary_percentage > 0:
                    # Canary: Route subset
                    # For batch prediction, routing whole batch or splitting?
                    # Usually routing request. Let's route full batch for simplicity here.
                    if random.random() < self.config.canary_percentage:
                        serving_mode = ServingMode.CANARY
                        # Swap results
                        challenger_result = self._predict_single(self.challenger_model, feature_array)
                        champion_result = challenger_result # Use challenger as result
                        challenger_latency = (time.time() - challenger_start) * 1000

            total_latency = (time.time() - start_time) * 1000
            
            metrics.prediction_latency.observe(total_latency / 1000)
            metrics.successful_predictions.inc()
            
            logger.log_event(
                'prediction_serving_completed',
                mode=serving_mode,
                champion=self.champion_model.version,
                rows=len(features),
                latency_ms=round(total_latency, 2)
            )
            
            return {
                'predictions': champion_result.predictions.tolist(),
                'confidence_scores': champion_result.confidence_scores.tolist() if champion_result.confidence_scores is not None else None,
                'model_version': champion_result.model_version,
                'serving_mode': serving_mode,
                'latency_ms': total_latency,
                'request_id': corr_id
            }
            
        except Exception as e:
            logger.log_error('prediction_failed', error=str(e), exc_info=True)
            metrics.failed_predictions.labels(error_type=type(e).__name__).inc()
            
            if self.config.enable_fallback and self.challenger_model:
                logger.log_event('attempting_fallback')
                fallback_res = self._predict_single(self.challenger_model, feature_array)
                return {
                     'predictions': fallback_res.predictions.tolist(),
                     'confidence_scores': fallback_res.confidence_scores.tolist(),
                     'model_version': fallback_res.model_version,
                     'serving_mode': ServingMode.FALLBACK,
                     'status': 'fallback_success'
                }
            raise

    def _predict_single(self, model: xgb.XGBClassifier, features: np.ndarray) -> PredictionResult:
        start = time.time()
        preds = model.predict(features)
        probs = model.predict_proba(features)[:, 1] # Class 1
        dur = (time.time() - start) * 1000
        return PredictionResult(
            predictions=preds,
            confidence_scores=probs,
            model_version=getattr(model, 'version', 'unknown'),
            latency_ms=dur
        )

    def _log_shadow_diff(self, champion: PredictionResult, challenger: PredictionResult, corr_id: str):
        # Compare minimal stats
        match_rate = np.mean(champion.predictions == challenger.predictions)
        logger.log_event(
            'shadow_comparison', 
            correlation_id=corr_id,
            match_rate=match_rate
        )

# Global Instance
MODEL_SERVER = None

def get_model_server() -> ModelServer:
    global MODEL_SERVER
    if MODEL_SERVER is None:
        registry = ModelRegistry()
        config = ServingConfig.from_env()
        MODEL_SERVER = ModelServer(registry, config)
    return MODEL_SERVER
