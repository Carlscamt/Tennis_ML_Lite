"""
Tournament Bracket Data Structures and Logic.

Handles tournament bracket construction, round progression,
and match organization for showdown simulations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import polars as pl


# Round name to number mapping (higher = later in tournament)
ROUND_ORDER = {
    "Round of 128": 1,
    "Round of 64": 2, 
    "Round of 32": 3,
    "Round of 16": 4,
    "Quarterfinal": 5,
    "Quarterfinals": 5,
    "QF": 5,
    "Semifinal": 6,
    "Semifinals": 6,
    "SF": 6,
    "Final": 7,
    "F": 7,
    # Alternative naming
    "1st Round": 1,
    "2nd Round": 2,
    "3rd Round": 3,
    "4th Round": 4,
}


@dataclass
class TournamentConfig:
    """Configuration for a tournament simulation."""
    name: str                    # "Australian Open"
    year: int                    # 2025
    draw_size: int = 128         # 128, 64, 32
    surface: str = "Hard"        # Surface type
    
    @property
    def num_rounds(self) -> int:
        """Calculate number of rounds based on draw size."""
        import math
        return int(math.log2(self.draw_size))
    
    def __str__(self) -> str:
        return f"{self.name} {self.year}"


@dataclass
class BracketMatch:
    """A single match in the tournament bracket."""
    match_id: str                       # Unique identifier
    round_name: str                     # "Round of 128", "Final"
    round_num: int                      # 1-7
    match_num: int                      # Position in bracket within round
    
    # Players
    player1_id: Optional[int] = None
    player1_name: str = "TBD"
    player2_id: Optional[int] = None
    player2_name: str = "TBD"
    
    # Model predictions
    model_winner_id: Optional[int] = None
    model_winner_name: str = ""
    model_confidence: float = 0.0
    
    # Actual results (None for future matches)
    actual_winner_id: Optional[int] = None
    actual_winner_name: str = ""
    
    # Comparison
    prediction_correct: Optional[bool] = None
    
    # Metadata
    event_id: Optional[int] = None
    odds_player1: Optional[float] = None
    odds_player2: Optional[float] = None
    
    def set_prediction(self, winner_id: int, winner_name: str, confidence: float):
        """Set the model's prediction for this match."""
        self.model_winner_id = winner_id
        self.model_winner_name = winner_name
        self.model_confidence = confidence
    
    def set_actual_result(self, winner_id: int, winner_name: str):
        """Set the actual match result and calculate if prediction was correct."""
        self.actual_winner_id = winner_id
        self.actual_winner_name = winner_name
        if self.model_winner_id is not None:
            self.prediction_correct = (self.model_winner_id == winner_id)
    
    @property
    def is_completed(self) -> bool:
        """Check if match has an actual result."""
        return self.actual_winner_id is not None
    
    @property
    def is_predicted(self) -> bool:
        """Check if match has a model prediction."""
        return self.model_winner_id is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "match_id": self.match_id,
            "round_name": self.round_name,
            "round_num": self.round_num,
            "match_num": self.match_num,
            "player1_id": self.player1_id,
            "player1_name": self.player1_name,
            "player2_id": self.player2_id,
            "player2_name": self.player2_name,
            "model_winner_id": self.model_winner_id,
            "model_winner_name": self.model_winner_name,
            "model_confidence": self.model_confidence,
            "actual_winner_id": self.actual_winner_id,
            "actual_winner_name": self.actual_winner_name,
            "prediction_correct": self.prediction_correct,
            "event_id": self.event_id,
        }


@dataclass
class TournamentBracket:
    """
    Complete tournament bracket structure.
    
    Organizes matches by round and provides methods for:
    - Building bracket from historical data
    - Running simulations with model predictions
    - Comparing predictions vs actual results
    """
    config: TournamentConfig
    rounds: Dict[int, List[BracketMatch]] = field(default_factory=dict)
    
    @property
    def all_matches(self) -> List[BracketMatch]:
        """Get all matches in bracket order (round 1 first)."""
        matches = []
        for round_num in sorted(self.rounds.keys()):
            matches.extend(self.rounds[round_num])
        return matches
    
    @property
    def total_matches(self) -> int:
        """Total number of matches in bracket."""
        return sum(len(matches) for matches in self.rounds.values())
    
    def get_round_matches(self, round_num: int) -> List[BracketMatch]:
        """Get all matches for a specific round."""
        return self.rounds.get(round_num, [])
    
    def get_match(self, round_num: int, match_num: int) -> Optional[BracketMatch]:
        """Get a specific match by round and position."""
        matches = self.rounds.get(round_num, [])
        for match in matches:
            if match.match_num == match_num:
                return match
        return None
    
    @classmethod
    def from_historical_data(
        cls, 
        df: pl.DataFrame, 
        tournament_name: str, 
        year: int
    ) -> "TournamentBracket":
        """
        Build tournament bracket from historical match data.
        
        Args:
            df: DataFrame with match data (must have tournament_name, round_name, etc.)
            tournament_name: Tournament to filter for
            year: Year to filter for
            
        Returns:
            TournamentBracket with all matches populated
        """
        # Filter to specific tournament and year
        start_of_year = datetime(year, 1, 1).timestamp()
        end_of_year = datetime(year, 12, 31, 23, 59, 59).timestamp()
        
        # Normalize tournament name for matching
        tournament_filter = tournament_name.lower()
        
        filtered = df.filter(
            (pl.col("tournament_name").str.to_lowercase().str.contains(tournament_filter)) &
            (pl.col("start_timestamp") >= start_of_year) &
            (pl.col("start_timestamp") <= end_of_year)
        )
        
        if len(filtered) == 0:
            raise ValueError(f"No matches found for {tournament_name} {year}")
        
        # Determine draw size from number of unique players
        # Count unique player_ids and opponent_ids separately, then combine
        player_ids = set(filtered.select("player_id").to_series().to_list())
        opponent_ids = set(filtered.select("opponent_id").to_series().to_list())
        unique_players = len(player_ids | opponent_ids)
        
        # Estimate draw size (round to power of 2)
        import math
        draw_size = 2 ** math.ceil(math.log2(max(unique_players, 2)))
        draw_size = min(max(draw_size, 32), 128)  # Clamp to reasonable range
        
        # Get surface from first match
        surface = filtered.select("ground_type").head(1).item() or "Hard"
        
        config = TournamentConfig(
            name=tournament_name,
            year=year,
            draw_size=draw_size,
            surface=surface
        )
        
        bracket = cls(config=config)
        
        # Group matches by round and build bracket
        # First, add round_num if not present
        if "round_num" not in filtered.columns:
            filtered = filtered.with_columns(
                pl.col("round_name").replace(ROUND_ORDER, default=3).alias("round_num")
            )
        
        # Process each match 
        matches_seen = set()  # Track unique matches (player pairs)
        
        for row in filtered.iter_rows(named=True):
            # Create a unique key for this matchup
            p1, p2 = sorted([row["player_id"], row["opponent_id"]])
            match_key = (row.get("event_id"), p1, p2)
            
            if match_key in matches_seen:
                continue
            matches_seen.add(match_key)
            
            round_name = row.get("round_name", "Unknown")
            round_num = ROUND_ORDER.get(round_name, 3)
            
            if round_num not in bracket.rounds:
                bracket.rounds[round_num] = []
            
            match_num = len(bracket.rounds[round_num]) + 1
            
            # Determine winner from data
            player_won = row.get("player_won")
            if player_won is True:
                winner_id = row["player_id"]
                winner_name = row.get("player_name", "Unknown")
            elif player_won is False:
                winner_id = row["opponent_id"]
                winner_name = row.get("opponent_name", "Unknown")
            else:
                winner_id = None
                winner_name = ""
            
            match = BracketMatch(
                match_id=f"R{round_num}M{match_num}",
                round_name=round_name,
                round_num=round_num,
                match_num=match_num,
                player1_id=row["player_id"],
                player1_name=row.get("player_name", "Player 1"),
                player2_id=row["opponent_id"],
                player2_name=row.get("opponent_name", "Player 2"),
                actual_winner_id=winner_id,
                actual_winner_name=winner_name,
                event_id=row.get("event_id"),
                odds_player1=row.get("odds_player"),
                odds_player2=row.get("odds_opponent"),
            )
            
            bracket.rounds[round_num].append(match)
        
        # Sort matches within each round by match number
        for round_num in bracket.rounds:
            bracket.rounds[round_num].sort(key=lambda m: m.match_num)
        
        return bracket
    
    def get_accuracy_by_round(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate prediction accuracy for each round.
        
        Returns:
            Dict mapping round name to accuracy stats
        """
        stats = {}
        
        for round_num in sorted(self.rounds.keys()):
            matches = self.rounds[round_num]
            predicted_matches = [m for m in matches if m.prediction_correct is not None]
            
            if not predicted_matches:
                continue
            
            correct = sum(1 for m in predicted_matches if m.prediction_correct)
            total = len(predicted_matches)
            
            # Get round name from first match
            round_name = matches[0].round_name if matches else f"Round {round_num}"
            
            stats[round_name] = {
                "round_num": round_num,
                "total_matches": total,
                "correct_predictions": correct,
                "accuracy": correct / total if total > 0 else 0.0,
                "avg_confidence": sum(m.model_confidence for m in predicted_matches) / total if total > 0 else 0.0,
            }
        
        return stats
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall tournament prediction statistics."""
        all_predicted = [m for m in self.all_matches if m.prediction_correct is not None]
        
        if not all_predicted:
            return {
                "total_matches": self.total_matches,
                "predicted_matches": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
            }
        
        correct = sum(1 for m in all_predicted if m.prediction_correct)
        
        return {
            "tournament": str(self.config),
            "total_matches": self.total_matches,
            "predicted_matches": len(all_predicted),
            "correct_predictions": correct,
            "accuracy": correct / len(all_predicted),
            "avg_confidence": sum(m.model_confidence for m in all_predicted) / len(all_predicted),
            "by_round": self.get_accuracy_by_round(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize bracket to dictionary."""
        return {
            "config": {
                "name": self.config.name,
                "year": self.config.year,
                "draw_size": self.config.draw_size,
                "surface": self.config.surface,
            },
            "rounds": {
                str(k): [m.to_dict() for m in v] 
                for k, v in self.rounds.items()
            },
            "stats": self.get_overall_stats(),
        }
