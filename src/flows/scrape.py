"""
Scraping flows - Data collection from SofaScore API.
"""
from prefect import flow, task
from prefect.logging import get_run_logger
from typing import List, Dict, Optional
import polars as pl


@task(retries=2, retry_delay_seconds=60, name="fetch-rankings")
def fetch_rankings_task(ranking_type: str = "atp_singles", top_n: int = 50) -> List[Dict]:
    """Fetch player rankings from API."""
    from src.scraper import fetch_rankings
    
    logger = get_run_logger()
    logger.info(f"Fetching {ranking_type} rankings, top {top_n}")
    
    players = fetch_rankings(ranking_type, top_n)
    logger.info(f"Found {len(players)} players")
    return players


@task(retries=2, retry_delay_seconds=30, name="fetch-player-matches")
def fetch_player_matches_task(player_id: int, pages: int = 10) -> List[Dict]:
    """Fetch historical matches for a player."""
    from src.scraper import fetch_player_matches
    
    logger = get_run_logger()
    matches = fetch_player_matches(player_id, max_pages=pages)
    logger.info(f"Player {player_id}: fetched {len(matches)} matches")
    return matches


@task(name="save-historical-data")
def save_historical_data_task(records: List[Dict], output_path: str) -> str:
    """Save scraped records to parquet."""
    from pathlib import Path
    
    logger = get_run_logger()
    
    if not records:
        logger.warning("No records to save")
        return ""
    
    df = pl.DataFrame(records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output, compression="snappy")
    
    logger.info(f"Saved {len(records)} records to {output}")
    return str(output)


@task(retries=2, retry_delay_seconds=60, name="fetch-upcoming-matches")
def fetch_upcoming_task(days: int = 7) -> List[Dict]:
    """Fetch upcoming matches."""
    from src.scraper import scrape_upcoming
    
    logger = get_run_logger()
    logger.info(f"Fetching upcoming matches for next {days} days")
    
    df = scrape_upcoming(days=days)
    records = df.to_dicts() if len(df) > 0 else []
    
    logger.info(f"Found {len(records)} upcoming matches")
    return records


@flow(name="scrape-historical", log_prints=True)
def scrape_historical_flow(
    top_players: int = 50,
    pages: int = 10,
    ranking_type: str = "atp_singles"
) -> str:
    """
    Scrape historical match data for top-ranked players.
    
    Args:
        top_players: Number of top players to scrape
        pages: Pages of matches per player
        ranking_type: atp_singles or wta_singles
        
    Returns:
        Path to saved parquet file
    """
    from pathlib import Path
    from datetime import datetime
    
    logger = get_run_logger()
    logger.info(f"Starting historical scrape: top {top_players}, {pages} pages per player")
    
    # Fetch rankings
    players = fetch_rankings_task(ranking_type, top_players)
    
    if not players:
        logger.error("Failed to fetch rankings")
        return ""
    
    # Fetch matches for each player
    all_records = []
    for player in players:
        player_id = player.get("id")
        if player_id:
            matches = fetch_player_matches_task(player_id, pages)
            all_records.extend(matches)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d")
    output_path = f"data/raw/historical_{ranking_type}_{timestamp}.parquet"
    
    result_path = save_historical_data_task(all_records, output_path)
    
    logger.info(f"Historical scrape complete: {len(all_records)} total matches")
    return result_path


@flow(name="scrape-upcoming", log_prints=True)
def scrape_upcoming_flow(days: int = 7) -> str:
    """
    Scrape upcoming matches for the next N days.
    
    Args:
        days: Number of days to look ahead
        
    Returns:
        Path to saved parquet file
    """
    from pathlib import Path
    
    logger = get_run_logger()
    logger.info(f"Starting upcoming scrape for {days} days")
    
    records = fetch_upcoming_task(days)
    
    if not records:
        logger.warning("No upcoming matches found")
        return ""
    
    output_path = "data/upcoming.parquet"
    result_path = save_historical_data_task(records, output_path)
    
    logger.info(f"Upcoming scrape complete: {len(records)} matches")
    return result_path
