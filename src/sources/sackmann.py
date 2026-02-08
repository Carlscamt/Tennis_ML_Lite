"""
Jeff Sackmann's tennis_atp data source adapter.

Fetches and parses CSV data from the tennis_atp GitHub repository:
https://github.com/JeffSackmann/tennis_atp

Data includes:
- ATP matches from 1968-present
- Rankings from 1985-present  
- Player biographical data
- Match statistics from 1991-present

License: CC BY-NC-SA 4.0 (Attribution, NonCommercial, ShareAlike)
"""
import logging
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from io import StringIO
import polars as pl

logger = logging.getLogger(__name__)


SACKMANN_BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"


@dataclass
class SackmannDataSource:
    """
    Adapter for Jeff Sackmann's tennis_atp CSV data.
    
    Fetches data directly from GitHub raw URLs.
    Includes local caching to reduce API calls.
    
    Example:
        source = SackmannDataSource()
        matches_2023 = source.fetch_matches(2023)
        players = source.fetch_players()
    """
    base_url: str = SACKMANN_BASE_URL
    cache_dir: Optional[Path] = None
    cache_ttl_hours: int = 24
    
    def __post_init__(self):
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _fetch_csv(self, filename: str) -> Optional[pl.DataFrame]:
        """Fetch CSV from GitHub and parse into DataFrame."""
        import requests
        
        url = f"{self.base_url}/{filename}"
        
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / filename
            if cache_path.exists():
                cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
                if cache_age < self.cache_ttl_hours * 3600:
                    logger.debug(f"Cache hit for {filename}")
                    return pl.read_csv(cache_path)
        
        try:
            logger.info(f"Fetching {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse CSV
            df = pl.read_csv(StringIO(response.text))
            
            # Cache if enabled
            if self.cache_dir:
                cache_path = self.cache_dir / filename
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.write_csv(cache_path)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {filename}: {e}")
            return None
    
    def fetch_matches(self, year: int, level: str = "tour") -> Optional[pl.DataFrame]:
        """
        Fetch ATP match data for a specific year.
        
        Args:
            year: Year to fetch (1968-present)
            level: Level of matches ("tour", "qual_chall", "futures")
            
        Returns:
            DataFrame with match data or None if not available
        """
        if level == "tour":
            filename = f"atp_matches_{year}.csv"
        elif level == "qual_chall":
            filename = f"atp_matches_qual_chall_{year}.csv"
        elif level == "futures":
            filename = f"atp_matches_futures_{year}.csv"
        else:
            raise ValueError(f"Unknown level: {level}")
        
        df = self._fetch_csv(filename)
        
        if df is not None:
            df = self._normalize_matches(df)
            logger.info(f"Fetched {len(df)} matches for {year} ({level})")
        
        return df
    
    def fetch_matches_range(
        self, 
        start_year: int, 
        end_year: int,
        level: str = "tour"
    ) -> pl.DataFrame:
        """Fetch matches for a range of years."""
        dfs = []
        
        for year in range(start_year, end_year + 1):
            df = self.fetch_matches(year, level)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            return pl.DataFrame()
        
        return pl.concat(dfs, how="diagonal_relaxed")
    
    def fetch_rankings(self, decade: str = "current") -> Optional[pl.DataFrame]:
        """
        Fetch ATP rankings data.
        
        Args:
            decade: Decade to fetch ("70s", "80s", "90s", "00s", "10s", "20s", "current")
            
        Returns:
            DataFrame with ranking data
        """
        filename = f"atp_rankings_{decade}.csv"
        df = self._fetch_csv(filename)
        
        if df is not None:
            df = self._normalize_rankings(df)
            logger.info(f"Fetched {len(df)} ranking entries for {decade}")
        
        return df
    
    def fetch_players(self) -> Optional[pl.DataFrame]:
        """
        Fetch player biographical data.
        
        Returns:
            DataFrame with player_id, name, hand, dob, country, height
        """
        df = self._fetch_csv("atp_players.csv")
        
        if df is not None:
            df = self._normalize_players(df)
            logger.info(f"Fetched {len(df)} player records")
        
        return df
    
    def _normalize_matches(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize match data to consistent schema."""
        # Rename to consistent column names
        rename_map = {
            "tourney_id": "tournament_id",
            "tourney_name": "tournament_name",
            "tourney_date": "match_date",
            "winner_id": "winner_player_id",
            "winner_name": "winner_name",
            "loser_id": "loser_player_id",
            "loser_name": "loser_name",
        }
        
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename({old: new})
        
        # Parse date if present
        if "match_date" in df.columns:
            df = df.with_columns(
                pl.col("match_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("match_date")
            )
        
        # Normalize surface
        if "surface" in df.columns:
            df = df.with_columns(
                pl.col("surface").str.to_lowercase().alias("surface")
            )
        
        return df
    
    def _normalize_rankings(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize ranking data."""
        if "ranking_date" in df.columns:
            df = df.with_columns(
                pl.col("ranking_date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("ranking_date")
            )
        
        return df
    
    def _normalize_players(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize player data."""
        if "dob" in df.columns:
            df = df.with_columns(
                pl.col("dob").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False).alias("birth_date")
            )
        
        # Create full name column
        if "name_first" in df.columns and "name_last" in df.columns:
            df = df.with_columns(
                (pl.col("name_first") + " " + pl.col("name_last")).alias("player_name")
            )
        
        return df
    
    def get_surface_mapping(self) -> Dict[str, str]:
        """Surface name mapping to canonical values."""
        return {
            "hard": "hard",
            "clay": "clay",
            "grass": "grass",
            "carpet": "carpet",
        }


def get_sackmann_source(cache_dir: Path = None) -> SackmannDataSource:
    """Get Sackmann data source with optional caching."""
    if cache_dir is None:
        cache_dir = Path("data/.cache/sackmann")
    return SackmannDataSource(cache_dir=cache_dir)
