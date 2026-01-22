"""
Configuration - All settings in one place
"""
from pathlib import Path

# Paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# API
SOFASCORE_URL = "https://www.sofascore.com/api/v1"

# Rate limiting
MIN_DELAY = 0.3
MAX_DELAY = 0.8

# Model
TEST_SPLIT_DATE = "2025-01-01"
RANDOM_STATE = 42

# Betting
MIN_ODDS = 1.5
MAX_ODDS = 3.0
MIN_EDGE = 0.05
KELLY_FRACTION = 0.25
