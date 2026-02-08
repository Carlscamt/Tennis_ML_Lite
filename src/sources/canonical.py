"""
Canonical schema for multi-source data reconciliation.

Defines a common schema that data from all sources can be mapped to,
enabling cross-source validation and anomaly detection.
"""
from datetime import date
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import polars as pl
import logging

logger = logging.getLogger(__name__)


class Surface(str, Enum):
    """Canonical surface types."""
    HARD = "hard"
    CLAY = "clay"
    GRASS = "grass"
    CARPET = "carpet"
    UNKNOWN = "unknown"


class DataSource(str, Enum):
    """Supported data sources."""
    SOFASCORE = "sofascore"
    SACKMANN = "sackmann"


@dataclass
class CanonicalMatch:
    """
    Canonical representation of a tennis match.
    
    Maps fields from different sources to a common schema.
    """
    # Required fields
    source: DataSource
    match_id: str  # Unique within source
    match_date: date
    
    # Player info (winner perspective)
    winner_id: int
    winner_name: str
    loser_id: int
    loser_name: str
    
    # Match context
    tournament_name: Optional[str] = None
    surface: Surface = Surface.UNKNOWN
    round: Optional[str] = None
    
    # Score
    score: Optional[str] = None
    sets_won: Optional[int] = None
    sets_lost: Optional[int] = None
    
    # Rankings at time of match
    winner_rank: Optional[int] = None
    loser_rank: Optional[int] = None
    
    # Odds (SofaScore only)
    winner_odds: Optional[float] = None
    loser_odds: Optional[float] = None
    
    # Raw source data for debugging
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "match_id": self.match_id,
            "match_date": self.match_date.isoformat() if self.match_date else None,
            "winner_id": self.winner_id,
            "winner_name": self.winner_name,
            "loser_id": self.loser_id,
            "loser_name": self.loser_name,
            "tournament_name": self.tournament_name,
            "surface": self.surface.value,
            "round": self.round,
            "score": self.score,
            "winner_rank": self.winner_rank,
            "loser_rank": self.loser_rank,
            "winner_odds": self.winner_odds,
            "loser_odds": self.loser_odds,
        }


def _parse_surface(value: str) -> Surface:
    """Parse surface string to canonical enum."""
    if not value:
        return Surface.UNKNOWN
    
    value = value.lower().strip()
    mapping = {
        "hard": Surface.HARD,
        "clay": Surface.CLAY,
        "grass": Surface.GRASS,
        "carpet": Surface.CARPET,
        "indoor hard": Surface.HARD,
        "outdoor hard": Surface.HARD,
    }
    return mapping.get(value, Surface.UNKNOWN)


def to_canonical_from_sackmann(df: pl.DataFrame) -> List[CanonicalMatch]:
    """
    Convert Sackmann DataFrame to canonical matches.
    
    Args:
        df: DataFrame from SackmannDataSource.fetch_matches()
        
    Returns:
        List of CanonicalMatch objects
    """
    matches = []
    
    for row in df.to_dicts():
        try:
            # Parse match date
            match_date_val = row.get("match_date")
            if isinstance(match_date_val, str):
                match_date_val = date.fromisoformat(match_date_val)
            elif not isinstance(match_date_val, date):
                match_date_val = None
            
            match = CanonicalMatch(
                source=DataSource.SACKMANN,
                match_id=f"sack_{row.get('tournament_id', '')}_{row.get('match_num', '')}",
                match_date=match_date_val,
                winner_id=row.get("winner_player_id", 0),
                winner_name=row.get("winner_name", ""),
                loser_id=row.get("loser_player_id", 0),
                loser_name=row.get("loser_name", ""),
                tournament_name=row.get("tournament_name"),
                surface=_parse_surface(row.get("surface", "")),
                round=row.get("round"),
                score=row.get("score"),
                winner_rank=row.get("winner_rank"),
                loser_rank=row.get("loser_rank"),
                raw_data=row,
            )
            matches.append(match)
            
        except Exception as e:
            logger.warning(f"Failed to convert Sackmann row: {e}")
    
    return matches


def to_canonical_from_sofascore(df: pl.DataFrame) -> List[CanonicalMatch]:
    """
    Convert SofaScore DataFrame to canonical matches.
    
    Args:
        df: DataFrame from SofaScore scraper
        
    Returns:
        List of CanonicalMatch objects
    """
    from datetime import datetime
    
    matches = []
    
    # Group by event_id to get both players from the same match
    if "event_id" not in df.columns:
        return matches
    
    for row in df.to_dicts():
        try:
            # Determine winner/loser based on player_won
            if row.get("player_won") is True:
                winner_id = row.get("player_id", 0)
                winner_name = row.get("player_name", "")
                loser_id = row.get("opponent_id", 0)
                loser_name = row.get("opponent_name", "")
                winner_odds = row.get("odds_player")
                loser_odds = row.get("odds_opponent")
            elif row.get("player_won") is False:
                winner_id = row.get("opponent_id", 0)
                winner_name = row.get("opponent_name", "")
                loser_id = row.get("player_id", 0)
                loser_name = row.get("player_name", "")
                winner_odds = row.get("odds_opponent")
                loser_odds = row.get("odds_player")
            else:
                continue  # Skip if outcome unknown
            
            # Parse timestamp
            ts = row.get("start_timestamp", 0)
            if ts:
                match_date = datetime.fromtimestamp(ts).date()
            else:
                match_date = None
            
            match = CanonicalMatch(
                source=DataSource.SOFASCORE,
                match_id=f"sofa_{row.get('event_id', '')}",
                match_date=match_date,
                winner_id=winner_id,
                winner_name=winner_name,
                loser_id=loser_id,
                loser_name=loser_name,
                tournament_name=row.get("tournament_name"),
                surface=_parse_surface(row.get("ground_type", "")),
                winner_odds=winner_odds,
                loser_odds=loser_odds,
                raw_data=row,
            )
            matches.append(match)
            
        except Exception as e:
            logger.warning(f"Failed to convert SofaScore row: {e}")
    
    return matches


def to_canonical(df: pl.DataFrame, source: DataSource) -> List[CanonicalMatch]:
    """
    Convert DataFrame to canonical matches based on source.
    
    Args:
        df: Source DataFrame
        source: Data source identifier
        
    Returns:
        List of CanonicalMatch objects
    """
    if source == DataSource.SACKMANN:
        return to_canonical_from_sackmann(df)
    elif source == DataSource.SOFASCORE:
        return to_canonical_from_sofascore(df)
    else:
        raise ValueError(f"Unknown source: {source}")


def canonical_to_dataframe(matches: List[CanonicalMatch]) -> pl.DataFrame:
    """Convert list of canonical matches to DataFrame."""
    if not matches:
        return pl.DataFrame()
    
    rows = [m.to_dict() for m in matches]
    return pl.DataFrame(rows)
