"""
Tournament Simulator - Match-by-match prediction engine.

Runs model predictions through tournament brackets,
simulating match outcomes round by round.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import polars as pl
import numpy as np
import logging

from .bracket import TournamentBracket, TournamentConfig, BracketMatch, ROUND_ORDER

logger = logging.getLogger(__name__)


@dataclass
class ShowdownStats:
    """Statistics from a tournament showdown."""
    tournament_name: str
    year: int
    total_matches: int
    predicted_matches: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    accuracy_by_round: Dict[str, float] = field(default_factory=dict)
    upsets_predicted: int = 0
    upsets_total: int = 0
    upset_accuracy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tournament": f"{self.tournament_name} {self.year}",
            "total_matches": self.total_matches,
            "predicted_matches": self.predicted_matches,
            "correct_predictions": self.correct_predictions,
            "accuracy": round(self.accuracy * 100, 1),
            "avg_confidence": round(self.avg_confidence * 100, 1),
            "accuracy_by_round": {
                k: round(v * 100, 1) for k, v in self.accuracy_by_round.items()
            },
            "upsets_predicted": self.upsets_predicted,
            "upsets_total": self.upsets_total,
            "upset_accuracy": round(self.upset_accuracy * 100, 1) if self.upsets_total > 0 else 0.0,
        }


class TournamentSimulator:
    """
    Simulate tournament brackets with model predictions.
    
    Loads tournament data, runs predictions through each round,
    and compares against actual results.
    """
    
    def __init__(
        self,
        data_path: Path,
        model_path: Optional[Path] = None,
        root_dir: Optional[Path] = None
    ):
        """
        Initialize simulator.
        
        Args:
            data_path: Path to historical match data (parquet)
            model_path: Path to trained model (optional, uses registry if None)
            root_dir: Project root directory
        """
        self.data_path = Path(data_path)
        self.model_path = model_path
        self.root_dir = root_dir or Path(__file__).parent.parent.parent
        
        self._predictor = None
        self._feature_engineer = None
        self._data = None
    
    @property
    def data(self) -> pl.DataFrame:
        """Lazily load data."""
        if self._data is None:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            self._data = pl.read_parquet(self.data_path)
            logger.info(f"Loaded {len(self._data):,} matches from {self.data_path}")
        return self._data
    
    @property
    def predictor(self):
        """Lazily load predictor."""
        if self._predictor is None:
            from src.model import Predictor, ModelRegistry
            
            if self.model_path:
                self._predictor = Predictor(self.model_path)
            else:
                # Use registry to find active model
                models_dir = self.root_dir / "models"
                registry = ModelRegistry(models_dir)
                active = registry.get_active_model()
                
                if not active:
                    raise RuntimeError("No active model found. Run training first.")
                
                self._predictor = Predictor(Path(active["path"]))
            
            logger.info(f"Loaded model for predictions")
        return self._predictor
    
    @property
    def feature_engineer(self):
        """Lazily load feature engineer."""
        if self._feature_engineer is None:
            from src.transform import FeatureEngineer
            self._feature_engineer = FeatureEngineer()
        return self._feature_engineer
    
    def list_available_tournaments(self, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List tournaments available in the data.
        
        Args:
            year: Optional year filter
            
        Returns:
            List of tournament info dicts
        """
        df = self.data
        
        if year:
            start_ts = datetime(year, 1, 1).timestamp()
            end_ts = datetime(year, 12, 31, 23, 59, 59).timestamp()
            df = df.filter(
                (pl.col("start_timestamp") >= start_ts) &
                (pl.col("start_timestamp") <= end_ts)
            )
        
        # Group by tournament
        tournaments = (
            df.group_by("tournament_name")
            .agg([
                pl.len().alias("match_count"),
                pl.col("start_timestamp").min().alias("first_match"),
                pl.col("start_timestamp").max().alias("last_match"),
            ])
            .sort("match_count", descending=True)
        )
        
        result = []
        for row in tournaments.iter_rows(named=True):
            if row["tournament_name"]:
                first_dt = datetime.fromtimestamp(row["first_match"])
                result.append({
                    "name": row["tournament_name"],
                    "matches": row["match_count"],
                    "year": first_dt.year,
                })
        
        return result
    
    def load_tournament_bracket(
        self, 
        tournament_name: str, 
        year: int
    ) -> TournamentBracket:
        """
        Load and build bracket from historical data.
        
        Args:
            tournament_name: Tournament to simulate
            year: Year of tournament
            
        Returns:
            TournamentBracket with actual results
        """
        logger.info(f"Loading bracket for {tournament_name} {year}")
        bracket = TournamentBracket.from_historical_data(
            self.data, tournament_name, year
        )
        logger.info(f"Built bracket with {bracket.total_matches} matches across {len(bracket.rounds)} rounds")
        return bracket
    
    def simulate_bracket(
        self, 
        bracket: TournamentBracket,
        use_actual_matchups: bool = True
    ) -> TournamentBracket:
        """
        Run model predictions through bracket.
        
        Args:
            bracket: Tournament bracket to simulate
            use_actual_matchups: If True, use actual match pairings (for comparison)
                                 If False, propagate predicted winners (pure simulation)
                                 
        Returns:
            Bracket with predictions filled in
        """
        logger.info(f"Simulating {bracket.config} with model predictions")
        
        # For historical comparison, predict each actual match
        for round_num in sorted(bracket.rounds.keys()):
            matches = bracket.rounds[round_num]
            
            for match in matches:
                if match.player1_id is None or match.player2_id is None:
                    continue
                
                try:
                    # Prepare match data for prediction
                    prediction = self._predict_match(
                        match.player1_id,
                        match.player1_name,
                        match.player2_id,
                        match.player2_name,
                        match.odds_player1,
                        match.odds_player2
                    )
                    
                    if prediction:
                        winner_id = prediction["winner_id"]
                        winner_name = prediction["winner_name"]
                        confidence = prediction["confidence"]
                        
                        match.set_prediction(winner_id, winner_name, confidence)
                        
                        # If actual result exists, calculate correctness
                        if match.actual_winner_id is not None:
                            match.prediction_correct = (match.model_winner_id == match.actual_winner_id)
                    
                except Exception as e:
                    logger.warning(f"Failed to predict match {match.match_id}: {e}")
                    continue
        
        return bracket
    
    def _predict_match(
        self,
        player1_id: int,
        player1_name: str,
        player2_id: int,
        player2_name: str,
        odds_player1: Optional[float] = None,
        odds_player2: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Make prediction for a single match.
        
        Returns:
            Dict with winner_id, winner_name, confidence or None
        """
        # Try model-based prediction first
        try:
            player1_stats = self._get_player_features(player1_id)
            player2_stats = self._get_player_features(player2_id)
            
            if player1_stats is not None and player2_stats is not None:
                features = self._build_prediction_features(
                    player1_stats, player2_stats,
                    odds_player1, odds_player2
                )
                
                if features is not None:
                    prob_player1 = self.predictor.predict_proba(features)
                    
                    if prob_player1 >= 0.5:
                        return {
                            "winner_id": player1_id,
                            "winner_name": player1_name,
                            "confidence": float(prob_player1),
                        }
                    else:
                        return {
                            "winner_id": player2_id,
                            "winner_name": player2_name,
                            "confidence": float(1 - prob_player1),
                        }
        except Exception as e:
            logger.debug(f"Model prediction failed, using odds fallback: {e}")
        
        # Fall back to odds-based prediction
        if odds_player1 and odds_player2 and odds_player1 > 0 and odds_player2 > 0:
            # Lower odds = favored
            if odds_player1 < odds_player2:
                return {
                    "winner_id": player1_id,
                    "winner_name": player1_name,
                    "confidence": 1 / odds_player1,
                }
            else:
                return {
                    "winner_id": player2_id,
                    "winner_name": player2_name,
                    "confidence": 1 / odds_player2,
                }
        
        return None
    
    
    def _get_player_features(self, player_id: int) -> Optional[Dict[str, float]]:
        """Get latest rolling features for a player."""
        player_matches = self.data.filter(pl.col("player_id") == player_id)
        
        if len(player_matches) == 0:
            return None
        
        # Get latest row (most recent features)
        latest = player_matches.sort("start_timestamp", descending=True).head(1)
        
        return latest.to_dicts()[0] if len(latest) > 0 else None
    
    def _build_prediction_features(
        self,
        player1_stats: Dict,
        player2_stats: Dict,
        odds_player1: Optional[float],
        odds_player2: Optional[float],
    ) -> Optional[np.ndarray]:
        """
        Build feature array for model prediction.
        
        Uses the same feature set as training (rolling stats, odds, etc.)
        """
        try:
            # Extract key features that the model expects
            features = {}
            
            # Odds features
            if odds_player1 and odds_player1 > 0:
                features["odds_player"] = odds_player1
                features["implied_prob_player"] = 1 / odds_player1
            else:
                features["odds_player"] = 2.0  # Default neutral odds
                features["implied_prob_player"] = 0.5
            
            # Rolling stats from player 1
            for key in ["player_win_rate_20", "player_win_rate_50", "player_rank_position"]:
                if key in player1_stats and player1_stats[key] is not None:
                    features[key] = player1_stats[key]
            
            # Head to head (if available)
            features["h2h_win_rate"] = 0.5  # Default no history
            
            # Surface features
            features["is_hard"] = 1 if player1_stats.get("ground_type", "").lower() == "hard" else 0
            features["is_clay"] = 1 if player1_stats.get("ground_type", "").lower() == "clay" else 0
            features["is_grass"] = 1 if player1_stats.get("ground_type", "").lower() == "grass" else 0
            
            # Underdog indicator
            features["is_underdog"] = 1 if odds_player1 and odds_player1 > 2.0 else 0
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature building failed: {e}")
            return None
    
    def run_showdown(
        self,
        tournament_name: str,
        year: int
    ) -> Tuple[TournamentBracket, ShowdownStats]:
        """
        Run full showdown simulation.
        
        Args:
            tournament_name: Tournament to simulate
            year: Year of tournament
            
        Returns:
            Tuple of (simulated bracket, statistics)
        """
        # Load and build bracket
        bracket = self.load_tournament_bracket(tournament_name, year)
        
        # Run predictions
        bracket = self.simulate_bracket(bracket)
        
        # Calculate statistics
        stats = self._calculate_stats(bracket)
        
        return bracket, stats
    
    def _calculate_stats(self, bracket: TournamentBracket) -> ShowdownStats:
        """Calculate comprehensive showdown statistics."""
        overall = bracket.get_overall_stats()
        by_round = bracket.get_accuracy_by_round()
        
        # Calculate upset statistics
        upsets_predicted = 0
        upsets_total = 0
        
        for match in bracket.all_matches:
            if match.odds_player1 and match.odds_player2 and match.actual_winner_id:
                # Determine underdog (higher odds = underdog)
                underdog_id = match.player1_id if match.odds_player1 > match.odds_player2 else match.player2_id
                
                # Was this an upset?
                if match.actual_winner_id == underdog_id:
                    upsets_total += 1
                    if match.model_winner_id == underdog_id:
                        upsets_predicted += 1
        
        return ShowdownStats(
            tournament_name=bracket.config.name,
            year=bracket.config.year,
            total_matches=overall.get("total_matches", 0),
            predicted_matches=overall.get("predicted_matches", 0),
            correct_predictions=overall.get("correct_predictions", 0),
            accuracy=overall.get("accuracy", 0.0),
            avg_confidence=overall.get("avg_confidence", 0.0),
            accuracy_by_round={
                k: v.get("accuracy", 0.0) for k, v in by_round.items()
            },
            upsets_predicted=upsets_predicted,
            upsets_total=upsets_total,
            upset_accuracy=upsets_predicted / upsets_total if upsets_total > 0 else 0.0,
        )
