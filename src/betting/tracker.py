"""
Betting performance tracking and P&L analysis.
"""
import polars as pl
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict
import json
import logging

logger = logging.getLogger(__name__)


class BettingTracker:
    """
    Track betting performance over time.
    Persists to JSON for resumability.
    """
    
    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Directory to store tracking data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.bets_file = self.data_dir / "bet_history.json"
        self.bets = self._load_bets()
    
    def _load_bets(self) -> List[Dict]:
        """Load existing bet history."""
        if self.bets_file.exists():
            with open(self.bets_file, "r") as f:
                return json.load(f)
        return []
    
    def _save_bets(self) -> None:
        """Save bet history."""
        with open(self.bets_file, "w") as f:
            json.dump(self.bets, f, indent=2, default=str)
    
    def record_bet(
        self,
        event_id: int,
        player_name: str,
        opponent_name: str,
        odds: float,
        stake: float,
        model_prob: float,
        edge: float,
        outcome: Optional[bool] = None,
        tournament: str = "",
    ) -> None:
        """
        Record a new bet.
        
        Args:
            event_id: Unique match identifier
            player_name: Player bet on
            opponent_name: Opponent
            odds: Decimal odds
            stake: Amount staked
            model_prob: Model's win probability
            edge: Edge over market
            outcome: True=won, False=lost, None=pending
            tournament: Tournament name
        """
        bet = {
            "id": len(self.bets) + 1,
            "event_id": event_id,
            "timestamp": datetime.now().isoformat(),
            "player_name": player_name,
            "opponent_name": opponent_name,
            "tournament": tournament,
            "odds": odds,
            "stake": stake,
            "model_prob": model_prob,
            "edge": edge,
            "outcome": outcome,
            "profit": None,
        }
        
        # Calculate profit if outcome known
        if outcome is not None:
            bet["profit"] = stake * (odds - 1) if outcome else -stake
        
        self.bets.append(bet)
        self._save_bets()
        
        logger.info(f"Recorded bet #{bet['id']}: {player_name} @ {odds}")
    
    def update_outcome(self, event_id: int, won: bool) -> None:
        """
        Update outcome for a pending bet.
        
        Args:
            event_id: Event ID to update
            won: Whether the bet won
        """
        for bet in self.bets:
            if bet["event_id"] == event_id and bet["outcome"] is None:
                bet["outcome"] = won
                bet["profit"] = bet["stake"] * (bet["odds"] - 1) if won else -bet["stake"]
                bet["settled_at"] = datetime.now().isoformat()
                self._save_bets()
                logger.info(f"Updated bet {event_id}: {'Won' if won else 'Lost'}")
                return
        
        logger.warning(f"No pending bet found for event {event_id}")
    
    def get_pending_bets(self) -> pl.DataFrame:
        """Get all unsettled bets."""
        pending = [b for b in self.bets if b["outcome"] is None]
        if not pending:
            return pl.DataFrame()
        return pl.DataFrame(pending)
    
    def get_performance_stats(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict:
        """
        Get performance statistics.
        
        Args:
            start_date: Filter start (inclusive)
            end_date: Filter end (inclusive)
            
        Returns:
            Dict with performance metrics
        """
        settled = [b for b in self.bets if b["outcome"] is not None]
        
        # Apply date filters
        if start_date:
            settled = [
                b for b in settled 
                if datetime.fromisoformat(b["timestamp"]).date() >= start_date
            ]
        if end_date:
            settled = [
                b for b in settled
                if datetime.fromisoformat(b["timestamp"]).date() <= end_date
            ]
        
        if not settled:
            return {"total_bets": 0}
        
        total_bets = len(settled)
        wins = sum(1 for b in settled if b["outcome"])
        total_profit = sum(b["profit"] for b in settled)
        total_staked = sum(b["stake"] for b in settled)
        
        # Calculate streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        temp_streak = 0
        
        for bet in settled:
            if bet["outcome"]:
                if temp_streak >= 0:
                    temp_streak += 1
                else:
                    temp_streak = 1
                max_win_streak = max(max_win_streak, temp_streak)
            else:
                if temp_streak <= 0:
                    temp_streak -= 1
                else:
                    temp_streak = -1
                max_loss_streak = max(max_loss_streak, -temp_streak)
        
        return {
            "total_bets": total_bets,
            "wins": wins,
            "losses": total_bets - wins,
            "win_rate": wins / total_bets,
            "total_profit": round(total_profit, 2),
            "total_staked": round(total_staked, 2),
            "roi": round(total_profit / total_staked * 100, 2) if total_staked else 0,
            "avg_odds": round(sum(b["odds"] for b in settled) / total_bets, 2),
            "avg_edge": round(sum(b["edge"] for b in settled) / total_bets * 100, 2),
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        }
    
    def get_daily_summary(self) -> pl.DataFrame:
        """Get daily P&L summary."""
        settled = [b for b in self.bets if b["outcome"] is not None]
        
        if not settled:
            return pl.DataFrame()
        
        df = pl.DataFrame(settled)
        
        return df.with_columns([
            pl.col("timestamp").str.slice(0, 10).alias("date")
        ]).group_by("date").agg([
            pl.len().alias("bets"),
            pl.col("outcome").sum().alias("wins"),
            pl.col("profit").sum().alias("profit"),
            pl.col("stake").sum().alias("staked"),
        ]).with_columns([
            (pl.col("profit") / pl.col("staked") * 100).round(1).alias("roi_pct"),
            pl.col("profit").cum_sum().alias("cumulative_profit"),
        ]).sort("date")
    
    def get_by_tournament(self) -> pl.DataFrame:
        """Get performance breakdown by tournament."""
        settled = [b for b in self.bets if b["outcome"] is not None and b.get("tournament")]
        
        if not settled:
            return pl.DataFrame()
        
        df = pl.DataFrame(settled)
        
        return df.group_by("tournament").agg([
            pl.len().alias("bets"),
            pl.col("outcome").sum().alias("wins"),
            pl.col("profit").sum().alias("profit"),
            pl.col("stake").sum().alias("staked"),
        ]).with_columns([
            (pl.col("wins") / pl.col("bets") * 100).round(1).alias("win_rate_pct"),
            (pl.col("profit") / pl.col("staked") * 100).round(1).alias("roi_pct"),
        ]).sort("profit", descending=True)
