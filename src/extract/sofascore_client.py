"""
SofaScore API client for fetching live data.
Used for real-time predictions and odds updates.
"""
import time
import random
from typing import Optional, Dict, List
from threading import Lock
import logging

logger = logging.getLogger(__name__)

try:
    from tls_client import Session
    TLS_AVAILABLE = True
except ImportError:
    TLS_AVAILABLE = False
    logger.warning("tls_client not available, using httpx fallback")
    import httpx


class SofaScoreClient:
    """
    Thread-safe SofaScore API client with rate limiting.
    Uses TLS fingerprint spoofing to bypass Cloudflare.
    """
    
    BASE_URL = "https://www.sofascore.com/api/v1"
    
    def __init__(self, max_retries: int = 3, delay_range: tuple = (0.3, 0.8)):
        """
        Initialize client.
        
        Args:
            max_retries: Number of retry attempts
            delay_range: Min/max delay between requests (seconds)
        """
        self.max_retries = max_retries
        self.delay_range = delay_range
        self.request_count = 0
        self._lock = Lock()
        
        if TLS_AVAILABLE:
            self.session = Session(client_identifier="firefox_120")
        else:
            self.session = httpx.Client(
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/120.0"}
            )
    
    def _fetch(self, endpoint: str) -> Optional[Dict]:
        """
        Fetch JSON from API with retries.
        
        Args:
            endpoint: API endpoint (e.g., '/rankings/5')
            
        Returns:
            JSON data or None if failed
        """
        url = f"{self.BASE_URL}{endpoint}" if endpoint.startswith("/") else endpoint
        
        for attempt in range(self.max_retries + 1):
            try:
                time.sleep(random.uniform(*self.delay_range))
                
                response = self.session.get(url)
                
                with self._lock:
                    self.request_count += 1
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 403:
                    # Rate limited
                    wait = 5 * (attempt + 1)
                    logger.warning(f"Rate limited, waiting {wait}s")
                    time.sleep(wait)
                elif response.status_code == 404:
                    return None
                    
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        
        return None
    
    def get_rankings(self, ranking_type: int = 5) -> List[Dict]:
        """
        Fetch current rankings.
        
        Args:
            ranking_type: 5=ATP Singles, 6=WTA Singles
            
        Returns:
            List of player ranking dicts
        """
        data = self._fetch(f"/rankings/{ranking_type}")
        
        if not data or "rankingRows" not in data:
            return []
        
        players = []
        for row in data["rankingRows"]:
            team = row.get("team", {})
            players.append({
                "position": row.get("position"),
                "player_id": team.get("id"),
                "name": team.get("name"),
                "country": team.get("country", {}).get("alpha2"),
                "points": row.get("points"),
            })
        
        return players
    
    def get_upcoming_matches(self, date_str: str = None) -> List[Dict]:
        """
        Fetch upcoming/today's matches.
        
        Args:
            date_str: Date in YYYY-MM-DD format, defaults to today
            
        Returns:
            List of upcoming match events
        """
        if date_str is None:
            from datetime import date
            date_str = date.today().isoformat()
        
        data = self._fetch(f"/sport/tennis/scheduled-events/{date_str}")
        
        if not data or "events" not in data:
            return []
        
        return data["events"]
    
    def get_match_odds(self, event_id: int) -> Dict:
        """
        Fetch odds for a specific match.
        
        Args:
            event_id: SofaScore event ID
            
        Returns:
            Dict with odds_home, odds_away
        """
        data = self._fetch(f"/event/{event_id}/odds/1/all")
        
        if not data or "markets" not in data:
            return {}
        
        odds = {}
        for market in data.get("markets", []):
            if market.get("marketId") == 1:  # Match Winner
                for choice in market.get("choices", []):
                    frac = choice.get("fractionalValue", "")
                    decimal = self._convert_fractional(frac)
                    if decimal:
                        if choice.get("name") == "1":
                            odds["odds_home"] = decimal
                        elif choice.get("name") == "2":
                            odds["odds_away"] = decimal
        
        return odds
    
    @staticmethod
    def _convert_fractional(frac_str: str) -> Optional[float]:
        """Convert fractional odds (e.g., '8/13') to decimal."""
        try:
            if "/" in str(frac_str):
                num, den = map(int, str(frac_str).split("/"))
                return round(1 + (num / den), 3)
            return float(frac_str)
        except (ValueError, ZeroDivisionError):
            return None
