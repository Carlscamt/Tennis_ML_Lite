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
# INSTANTIATE DEFAULTS
# =============================================================================

FEATURES = FeatureConfig()
MODEL = ModelConfig()
BETTING = BettingConfig()
API = APIConfig()
