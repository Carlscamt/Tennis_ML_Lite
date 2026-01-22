"""
Scraper - SofaScore API data collection
"""
import time
import random
import httpx
import polars as pl
from datetime import datetime, timedelta
from config import SOFASCORE_URL, DATA_DIR, MIN_DELAY, MAX_DELAY

def fetch_json(endpoint: str) -> dict:
    """Fetch JSON from API with rate limiting."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = httpx.get(f"{SOFASCORE_URL}{endpoint}", headers=headers, timeout=30)
        return resp.json() if resp.status_code == 200 else {}
    except:
        return {}

def get_scheduled_events(date_str: str) -> list:
    """Get tennis events for a date."""
    data = fetch_json(f"/sport/tennis/scheduled-events/{date_str}")
    return data.get("events", [])

def is_valid_event(event: dict) -> bool:
    """Check if event is ATP/Challenger singles."""
    cat = event.get("tournament", {}).get("category", {})
    cat_slug = cat.get("slug", "").lower()
    tournament = event.get("tournament", {}).get("name", "").lower()
    
    # Must be ATP or Challenger
    if not ("atp" in cat_slug or "challenger" in cat_slug):
        return False
    
    # No doubles
    home = event.get("homeTeam", {})
    if home.get("type") == "doubles" or "/" in home.get("name", ""):
        return False
    
    return True

def fetch_match_odds(event_id: int) -> dict:
    """Fetch odds for a match."""
    data = fetch_json(f"/event/{event_id}/odds/1/all")
    markets = data.get("markets", [])
    
    for market in markets:
        if market.get("marketName") == "Full time":
            choices = market.get("choices", [])
            if len(choices) >= 2:
                return {
                    "home_odds": choices[0].get("fractionalValue", ""),
                    "away_odds": choices[1].get("fractionalValue", ""),
                }
    return {}

def scrape_upcoming(days: int = 7) -> pl.DataFrame:
    """Scrape upcoming matches."""
    print(f"Scraping upcoming matches ({days} days)...")
    
    records = []
    today = datetime.now()
    
    for i in range(days):
        date_str = (today + timedelta(days=i)).strftime("%Y-%m-%d")
        events = get_scheduled_events(date_str)
        
        for event in events:
            if not is_valid_event(event):
                continue
            
            home = event.get("homeTeam", {})
            away = event.get("awayTeam", {})
            odds = fetch_match_odds(event.get("id"))
            
            records.append({
                "event_id": event.get("id"),
                "date": date_str,
                "player_name": home.get("name"),
                "opponent_name": away.get("name"),
                "player_id": home.get("id"),
                "opponent_id": away.get("id"),
                "tournament": event.get("tournament", {}).get("name"),
                "odds_player": float(odds.get("home_odds") or 0),
                "odds_opponent": float(odds.get("away_odds") or 0),
            })
    
    df = pl.DataFrame(records)
    output = DATA_DIR / "upcoming.parquet"
    df.write_parquet(output)
    print(f"Saved {len(df)} matches to {output}")
    return df

if __name__ == "__main__":
    scrape_upcoming(7)
