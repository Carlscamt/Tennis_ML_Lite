"""
SQLite-based bankroll ledger for tracking open bets and effective bankroll.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OpenBet:
    """Represents an open (unsettled) bet."""
    id: int
    event_id: str
    stake: float
    odds: float
    model_prob: float
    model_version: Optional[str]
    placed_at: datetime


@dataclass
class SettledBet:
    """Represents a settled bet."""
    id: int
    event_id: str
    stake: float
    odds: float
    won: bool
    pnl: float
    placed_at: datetime
    settled_at: datetime


class BankrollLedger:
    """
    SQLite-based bankroll ledger with open bet tracking.
    
    Tracks:
    - Current bankroll
    - Open bets (pending settlement)
    - Effective bankroll (bankroll - open exposure)
    - Bet history with P&L
    """
    
    def __init__(self, db_path: str = "data/bankroll.db"):
        """
        Initialize ledger.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bankroll (
                    id INTEGER PRIMARY KEY,
                    amount REAL NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    stake REAL NOT NULL,
                    odds REAL NOT NULL,
                    model_prob REAL,
                    model_version TEXT,
                    placed_at TEXT NOT NULL,
                    settled_at TEXT,
                    won INTEGER,
                    pnl REAL,
                    UNIQUE(event_id)
                )
            """)
            
            # Initialize bankroll if not exists
            cursor = conn.execute("SELECT COUNT(*) FROM bankroll")
            if cursor.fetchone()[0] == 0:
                conn.execute(
                    "INSERT INTO bankroll (id, amount, updated_at) VALUES (1, 1000.0, ?)",
                    (datetime.now().isoformat(),)
                )
            
            conn.commit()
    
    def get_bankroll(self) -> float:
        """Get current bankroll."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT amount FROM bankroll WHERE id = 1")
            row = cursor.fetchone()
            return row[0] if row else 0.0
    
    def set_bankroll(self, amount: float) -> None:
        """Set current bankroll."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE bankroll SET amount = ?, updated_at = ? WHERE id = 1",
                (amount, datetime.now().isoformat())
            )
            conn.commit()
        logger.info(f"Bankroll set to {amount:.2f}")
    
    def get_open_bets(self) -> List[OpenBet]:
        """Get all open (unsettled) bets."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, event_id, stake, odds, model_prob, model_version, placed_at
                FROM bets
                WHERE won IS NULL
                ORDER BY placed_at DESC
            """)
            
            return [
                OpenBet(
                    id=row[0],
                    event_id=row[1],
                    stake=row[2],
                    odds=row[3],
                    model_prob=row[4],
                    model_version=row[5],
                    placed_at=datetime.fromisoformat(row[6])
                )
                for row in cursor.fetchall()
            ]
    
    def get_open_exposure(self) -> float:
        """Get total exposure from open bets."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COALESCE(SUM(stake), 0)
                FROM bets
                WHERE won IS NULL
            """)
            return cursor.fetchone()[0]
    
    def get_effective_bankroll(self) -> float:
        """
        Get effective bankroll for Kelly calculations.
        
        Effective = Current Bankroll - Open Exposure
        
        This prevents over-betting when many bets are open.
        """
        bankroll = self.get_bankroll()
        exposure = self.get_open_exposure()
        effective = bankroll - exposure
        return max(0, effective)
    
    def place_bet(
        self,
        event_id: str,
        stake: float,
        odds: float,
        model_prob: Optional[float] = None,
        model_version: Optional[str] = None
    ) -> int:
        """
        Record a new bet.
        
        Args:
            event_id: Unique event identifier
            stake: Amount staked
            odds: Decimal odds
            model_prob: Model's win probability
            model_version: Model version used
            
        Returns:
            Bet ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO bets (event_id, stake, odds, model_prob, model_version, placed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                stake,
                odds,
                model_prob,
                model_version,
                datetime.now().isoformat()
            ))
            conn.commit()
            bet_id = cursor.lastrowid
        
        logger.info(f"Placed bet #{bet_id}: {stake:.2f} @ {odds:.2f}")
        return bet_id
    
    def settle_bet(self, event_id: str, won: bool) -> float:
        """
        Settle a bet and update bankroll.
        
        Args:
            event_id: Event ID to settle
            won: Whether the bet won
            
        Returns:
            P&L from the bet
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get bet details
            cursor = conn.execute("""
                SELECT id, stake, odds FROM bets
                WHERE event_id = ? AND won IS NULL
            """, (event_id,))
            
            row = cursor.fetchone()
            if not row:
                logger.warning(f"No open bet found for event {event_id}")
                return 0.0
            
            bet_id, stake, odds = row
            
            # Calculate P&L
            if won:
                pnl = stake * (odds - 1)
            else:
                pnl = -stake
            
            # Update bet
            conn.execute("""
                UPDATE bets
                SET won = ?, pnl = ?, settled_at = ?
                WHERE id = ?
            """, (1 if won else 0, pnl, datetime.now().isoformat(), bet_id))
            
            # Update bankroll
            current = self.get_bankroll()
            new_bankroll = current + pnl
            conn.execute(
                "UPDATE bankroll SET amount = ?, updated_at = ? WHERE id = 1",
                (new_bankroll, datetime.now().isoformat())
            )
            
            conn.commit()
        
        logger.info(f"Settled bet for {event_id}: {'Won' if won else 'Lost'} ({pnl:+.2f})")
        return pnl
    
    def get_stats(self) -> dict:
        """Get betting statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Open bets
            open_cursor = conn.execute("""
                SELECT COUNT(*), COALESCE(SUM(stake), 0)
                FROM bets WHERE won IS NULL
            """)
            open_count, open_exposure = open_cursor.fetchone()
            
            # Settled bets
            settled_cursor = conn.execute("""
                SELECT 
                    COUNT(*),
                    SUM(CASE WHEN won = 1 THEN 1 ELSE 0 END),
                    COALESCE(SUM(pnl), 0),
                    COALESCE(SUM(stake), 0)
                FROM bets WHERE won IS NOT NULL
            """)
            total, wins, total_pnl, total_staked = settled_cursor.fetchone()
            
            return {
                "bankroll": self.get_bankroll(),
                "effective_bankroll": self.get_effective_bankroll(),
                "open_bets": open_count,
                "open_exposure": open_exposure,
                "settled_bets": total or 0,
                "wins": wins or 0,
                "win_rate": (wins / total) if total else 0,
                "total_pnl": total_pnl or 0,
                "roi": (total_pnl / total_staked * 100) if total_staked else 0,
            }
