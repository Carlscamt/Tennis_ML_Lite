"""
Global configuration for Tennis Betting ML Pipeline.
"""
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required, will use system env vars


# =============================================================================
# ENVIRONMENT
# =============================================================================

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

ENV = Environment(os.getenv("ENVIRONMENT", "development"))


# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
BACKTEST_DIR = DATA_DIR / "backtest"


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    # Rolling window sizes
    rolling_windows: tuple = (5, 10, 20, 50)
    
    # Minimum matches for reliable stats
    min_matches_for_stats: int = 5
    
    # ELO parameters
    elo_k_factor: float = 32.0
    elo_initial: float = 1500.0
    
    # Surface-specific ELO
    surfaces: tuple = ("Hard", "Clay", "Grass", "Carpet")


# =============================================================================
# MODEL
# =============================================================================

@dataclass
class ModelConfig:
    """Model training parameters."""
    # Train/test split
    train_cutoff_date: date = date(2025, 1, 1)
    test_start_date: date = date(2025, 1, 1)
    
    # Minimum data requirements
    min_training_samples: int = 10000
    min_odds_coverage: float = 0.7  # 70% of matches need odds
    
    # XGBoost defaults
    xgb_params: dict = None
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 6,
                "learning_rate": 0.05,
                "n_estimators": 500,
                "early_stopping_rounds": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }


# =============================================================================
# BETTING
# =============================================================================

@dataclass
class BettingConfig:
    """Betting strategy parameters."""
    # Kelly criterion
    kelly_fraction: float = 0.25  # 1/4 Kelly for safety
    
    # Edge thresholds
    min_edge: float = 0.05  # 5% minimum edge to bet
    min_confidence: float = 0.55  # Minimum model probability
    
    # Stake limits
    max_stake_pct: float = 0.03  # 3% max per bet
    min_stake_units: float = 0.5  # Minimum 0.5 units
    
    # Odds filters
    min_odds: float = 1.20
    max_odds: float = 5.00
    
    # Bankroll
    initial_bankroll: float = 1000.0
    
    # Uncertainty thresholds
    min_margin: float = float(os.getenv("BETTING_MIN_MARGIN", "0.10"))
    max_entropy: float = float(os.getenv("BETTING_MAX_ENTROPY", "0.65"))
    use_uncertainty_filter: bool = os.getenv("BETTING_USE_UNCERTAINTY_FILTER", "true").lower() == "true"
    
    # Calibration settings
    calibration_method: str = os.getenv("CALIBRATION_METHOD", "isotonic")  # isotonic, platt, ensemble
    group_calibration: bool = os.getenv("GROUP_CALIBRATION", "true").lower() == "true"
    min_calibration_samples: int = int(os.getenv("MIN_CALIBRATION_SAMPLES", "500"))


# =============================================================================
# API
# =============================================================================

@dataclass
class APIConfig:
    """API configuration parameters."""
    base_url: str = os.getenv("SOFASCORE_BASE_URL", "https://www.sofascore.com/api/v1")
    timeout: int = int(os.getenv("API_TIMEOUT", "30"))
    rate_limit_per_min: int = int(os.getenv("API_RATE_LIMIT_PER_MIN", "60"))
    retry_attempts: int = int(os.getenv("API_RETRY_ATTEMPTS", "3"))

SOFASCORE_BASE_URL = os.getenv("SOFASCORE_BASE_URL", "https://www.sofascore.com/api/v1")

RANKING_IDS = {
    "atp_singles": 5,
    "wta_singles": 6,
    "atp_doubles": 7,
    "wta_doubles": 8,
}


# =============================================================================
# DATA QUALITY
# =============================================================================

@dataclass
class DataQualityConfig:
    """Data quality thresholds."""
    # Staleness check
    stale_hours_warn: float = float(os.getenv("STALE_HOURS_WARN", "24.0"))
    stale_hours_error: float = float(os.getenv("STALE_HOURS_ERROR", "48.0"))
    
    # Schema validation
    min_rows_warn: int = 10
    max_null_pct: float = 0.5


# =============================================================================
# SCRAPER CONFIGURATION
# =============================================================================

@dataclass
class ScraperConfig:
    """Scraper rate limiting configuration (tunable via environment variables)."""
    min_delay: float = float(os.getenv("SCRAPER_MIN_DELAY", "1.5"))
    max_delay: float = float(os.getenv("SCRAPER_MAX_DELAY", "3.0"))
    max_workers: int = int(os.getenv("SCRAPER_MAX_WORKERS", "2"))
    cache_ttl_rankings: int = int(os.getenv("CACHE_TTL_RANKINGS", "86400"))   # 24h
    cache_ttl_matches: int = int(os.getenv("CACHE_TTL_MATCHES", "3600"))      # 1h
    cache_ttl_odds: int = int(os.getenv("CACHE_TTL_ODDS", "900"))             # 15min


# =============================================================================
# CROSS-VALIDATION CONFIGURATION
# =============================================================================

@dataclass
class CrossValidationConfig:
    """Time-series cross-validation parameters."""
    n_splits: int = int(os.getenv("CV_N_SPLITS", "5"))
    gap_days: int = int(os.getenv("CV_GAP_DAYS", "7"))
    min_train_size: int = int(os.getenv("CV_MIN_TRAIN_SIZE", "5000"))
    rolling_window_days: int = int(os.getenv("CV_ROLLING_WINDOW_DAYS", "0"))  # 0 = expanding
    
    @property
    def use_rolling(self) -> bool:
        """True if using rolling window, False for expanding."""
        return self.rolling_window_days > 0


# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

@dataclass
class OptimizationConfig:
    """Hyperparameter optimization settings for Optuna."""
    n_trials: int = int(os.getenv("OPTUNA_N_TRIALS", "50"))
    timeout_seconds: int = int(os.getenv("OPTUNA_TIMEOUT", "3600"))
    variance_penalty: float = float(os.getenv("OPTUNA_VARIANCE_PENALTY", "0.1"))
    objective: str = os.getenv("OPTUNA_OBJECTIVE", "composite")  # composite, log_loss, roi
    n_jobs: int = int(os.getenv("OPTUNA_N_JOBS", "1"))
    storage: str = os.getenv("OPTUNA_STORAGE", "")  # e.g., sqlite:///optuna.db


# =============================================================================
# MODEL PROMOTION CONFIGURATION
# =============================================================================

@dataclass
class ModelPromotionConfig:
    """Thresholds for model promotion to Production."""
    min_auc: float = float(os.getenv("PROMOTION_MIN_AUC", "0.80"))
    min_sharpe: float = float(os.getenv("PROMOTION_MIN_SHARPE", "0.0"))
    require_positive_roi: bool = os.getenv("PROMOTION_REQUIRE_POSITIVE_ROI", "false").lower() == "true"


# =============================================================================
# INSTANTIATE DEFAULTS
# =============================================================================

FEATURES = FeatureConfig()
MODEL = ModelConfig()
BETTING = BettingConfig()
API = APIConfig()
SCRAPER = ScraperConfig()
DATA_QUALITY = DataQualityConfig()
CV = CrossValidationConfig()
PROMOTION = ModelPromotionConfig()
OPTIMIZATION = OptimizationConfig()
