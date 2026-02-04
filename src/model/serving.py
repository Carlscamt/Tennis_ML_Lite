
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
        """Load config from environment variables or config file (Env overrides file)."""
        defaults = {
            "canary_percentage": 0.0,
            "shadow_mode": False,
            "enable_fallback": True
        }
        
        # 1. Load from file
        config_path = "config/serving.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    defaults.update(file_config)
            except Exception as e:
                logger.log_error("failed_load_serving_config", error=str(e))
        
        # 2. Override with Env Vars
        canary = os.getenv('CANARY_PERCENTAGE')
        if canary is not None:
             defaults["canary_percentage"] = float(canary)
             
        shadow = os.getenv('SHADOW_MODE')
        if shadow is not None:
             defaults["shadow_mode"] = shadow.lower() == 'true'
             
        fallback = os.getenv('ENABLE_FALLBACK')
        if fallback is not None:
             defaults["enable_fallback"] = fallback.lower() == 'true'
        
        return cls(**defaults)

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
    
    def _load_model_artifact(self, path: str) -> Any:
        """Load model from path (supporting joblib or native xgb)."""
        import joblib
        path_obj = os.path.normpath(path)
        if path_obj.endswith('.joblib') or path_obj.endswith('.pkl'):
             try:
                 return joblib.load(path)
             except Exception as e:
                 logger.log_error("joblib_load_failed", path=path, error=str(e))
                 raise e
        else:
             # Assume native XGBoost config/binary
             # But if it fails (e.g. it's actually a pickle named .bin), try joblib
             try:
                 model = xgb.XGBClassifier()
                 model.load_model(path)
                 return model
             except Exception as e:
                 # XGBoost load failed (could be format error or "UnicodeDecodeError" in error msg)
                 # Fallback to joblib
                 try:
                     # logger.log_event("fallback_to_joblib_load", path=path, original_error=str(e))
                     return joblib.load(path)
                 except Exception:
                     # If both fail, raise the original XGBoost error as it's the expected format for unknown ext
                     raise e

    def _load_models(self):
        """Load champion (Production) and challenger (Staging) models."""
        success = False
        retries = 3
        for i in range(retries):
            try:
                # Load champion (always required)
                try:
                    champion_version, champion_path = self.registry.get_production_model()
                    self.champion_model = self._load_model_artifact(champion_path)
                    # Monkey-patch version for tracking
                    self.champion_model.version = champion_version
                    logger.log_event('champion_model_loaded', version=champion_version, attempt=i+1)
                except Exception as e:
                    # If strictly required, raise. But for daily batch, maybe we want to continue?
                    # The batch job will fail later if no model, but properly logged.
                    # Or we can try to find ANY model (Experimental) if config allows?
                    # For now just log and continue to allow generic init
                    logger.log_warning('production_model_load_failed', error=str(e))
                    if i == retries - 1:
                        # On last try, maybe try to load ANY model?
                        # For now, let's just not crash the constructor.
                        pass
                
                # Load challenger (optional)
                challenger = self.registry.get_challenger_model()
                if challenger:
                    challenger_version, challenger_path = challenger
                    self.challenger_model = self._load_model_artifact(challenger_path)
                    self.challenger_model.version = challenger_version
                    logger.log_event('challenger_model_loaded', version=challenger_version)
                else:
                    logger.log_event('no_challenger_model')
                
                # Desperation Mode: If no champion and no challenger, try LATEST (Experimental)
                if not self.champion_model and not self.challenger_model:
                     latest = self.registry.get_latest_model()
                     if latest:
                         v, p = latest
                         self.champion_model = self._load_model_artifact(p)
                         self.champion_model.version = v
                         logger.log_warning('SERVING_IN_EXPERIMENTAL_MODE', version=v, reason="No Production/Staging models found")
                
                success = True
                break # Success
            except Exception as e:
                logger.log_event('model_loading_attempt_failed', attempt=i+1, error=str(e), level='warning')
                if i < retries - 1:
                    time.sleep(0.5)
                    # Force registry reload to catch up with disk
                    if hasattr(self.registry, 'reload'):
                        self.registry.reload()
                    elif hasattr(self.registry, '_load_registry'):
                         # Fallback if reload not exposed
                         self.registry._load_registry()
                else:
                    logger.log_error('model_loading_final_failure', error=str(e), exc_info=True)
                    # Allow init to finish even if failed (predict will raise later)
                    pass

    def reload_models(self):
        """Force reload of models from registry."""
        logger.log_event('reloading_models_triggered')
        # Reload registry first to get latest metadata
        # if hasattr(self.registry, 'reload'):
        #     self.registry.reload()
        self._load_models()
    
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
        if not features:
            return {'predictions': []}
            
        corr_id = CORRELATION_ID.get() or request_id or str(random.randint(100000, 999999))
        CORRELATION_ID.set(corr_id)
        
        # Prepare Features (Dict List -> Numpy)
        feature_keys = list(features[0].keys())
        feature_array = np.array([[row[k] for k in feature_keys] for row in features])

        # Auto-fallback to challenger if no champion (e.g. freshly trained model in Staging)
        active_model = self.champion_model
        serving_mode = ServingMode.CHAMPION_ONLY
        
        if not active_model:
            if self.challenger_model:
                 logger.log_event('using_challenger_as_primary', version=self.challenger_model.version)
                 active_model = self.challenger_model
                 serving_mode = ServingMode.FALLBACK
            else:
                raise RuntimeError("Model Server not initialized with Production model")
        
        start_time = time.time()
        corr_id = str(random.randint(100000, 999999))
        champion_result = None
        challenger_result = None
        
        # Convert None to np.nan ONCE before any predictions
        # XGBoost can handle nan but not Python None
        import pandas as pd
        feature_array = pd.DataFrame(feature_array).fillna(np.nan).values.astype(np.float64)
        
        try:
            # 1. Predict with Active Model (Champion or Fallback)
            model_start = time.time()
            primary_result = self._predict_single(active_model, feature_array)
            
            # If we are already in FALLBACK mode, we are done
            if serving_mode == ServingMode.FALLBACK:
                total_latency = (time.time() - start_time) * 1000
                metrics.prediction_latency.observe(total_latency / 1000)
                metrics.successful_predictions.inc()
                
                return {
                    'predictions': primary_result.predictions.tolist(),
                    'confidence_scores': primary_result.confidence_scores.tolist() if primary_result.confidence_scores is not None else None,
                    'model_version': primary_result.model_version,
                    'serving_mode': serving_mode,
                    'latency_ms': total_latency,
                    'request_id': corr_id
                }

            # 2. Advanced Serving (Canary/Shadow) - Only if using Champion
            # (If we are here, active_model IS champion_model)
            champion_result = primary_result
            
            if self.challenger_model:
                challenger_start = time.time()
                
                if self.config.shadow_mode:
                    # Shadow: Run and log, discard result
                    serving_mode = ServingMode.SHADOW
                    challenger_result = self._predict_single(self.challenger_model, feature_array)
                    self._log_shadow_diff(champion_result, challenger_result, corr_id)
                    challenger_result = None 
                
                elif self.config.canary_percentage > 0:
                    if random.random() < self.config.canary_percentage:
                        serving_mode = ServingMode.CANARY
                        challenger_result = self._predict_single(self.challenger_model, feature_array)
                        primary_result = challenger_result # Swap result
            
            total_latency = (time.time() - start_time) * 1000
            
            metrics.prediction_latency.observe(total_latency / 1000)
            metrics.successful_predictions.inc()
            
            logger.log_event(
                'prediction_serving_completed',
                mode=serving_mode,
                champion=self.champion_model.version if self.champion_model else "none",
                rows=len(features),
                latency_ms=round(total_latency, 2)
            )
            
            return {
                'predictions': primary_result.predictions.tolist(),
                'confidence_scores': primary_result.confidence_scores.tolist() if primary_result.confidence_scores is not None else None,
                'model_version': primary_result.model_version,
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
        # Features are already cleaned (None -> np.nan) in _predict_sync
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
