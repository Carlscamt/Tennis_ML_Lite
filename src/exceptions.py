"""
Custom exceptions for Tennis Betting ML Pipeline.
"""


class TennisBettingError(Exception):
    """Base exception for all custom errors."""
    pass


# Data & Scraping Errors
class DataScrapingError(TennisBettingError):
    """Raised when data scraping fails."""
    pass


class InsufficientDataError(TennisBettingError):
    """Raised when not enough historical data is available."""
    def __init__(self, player_id: int = None, min_required: int = None):
        self.player_id = player_id
        self.min_required = min_required
        msg = f"Insufficient data"
        if player_id:
            msg += f" for player {player_id}"
        if min_required:
            msg += f" (minimum {min_required} matches required)"
        super().__init__(msg)


class APIRateLimitError(DataScrapingError):
    """Raised when API rate limit is exceeded."""
    pass


class DataValidationError(TennisBettingError):
    """Raised when data validation fails."""
    pass


# Model Errors
class ModelError(TennisBettingError):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when no active model is found."""
    pass


class ModelPredictionError(ModelError):
    """Raised when prediction generation fails."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


# Feature Engineering Errors
class FeatureEngineeringError(TennisBettingError):
    """Raised when feature engineering fails."""
    pass


# Betting Errors
class BettingError(TennisBettingError):
    """Base exception for betting-related errors."""
    pass


class InvalidStakeError(BettingError):
    """Raised when stake calculation is invalid."""
    pass


class InsufficientBankrollError(BettingError):
    """Raised when bankroll is too low for bet."""
    pass


# Configuration Errors
class ConfigurationError(TennisBettingError):
    """Raised when configuration is invalid or missing."""
    pass
