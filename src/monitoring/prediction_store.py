"""
Prediction persistence store.

SQLite-backed storage for predictions and outcomes, enabling:
- Next-day outcome matching
- ROI computation by model version
- Calibration analysis
"""
import sqlite3
import logging
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """A single prediction record."""
    event_id: int
    model_version: str
    predicted_prob: float
    predicted_outcome: bool  # True = player wins
    odds: float
    stake: float = 0.0
    surface: Optional[str] = None
    tournament: Optional[str] = None
    created_at: Optional[datetime] = None
    
    # Outcome fields (filled after match completes)
    actual_outcome: Optional[bool] = None
    pnl: Optional[float] = None
    resolved_at: Optional[datetime] = None


@dataclass
class PredictionStore:
    """
    SQLite store for prediction tracking and outcome matching.
    
    Schema:
    - predictions table: Stores all predictions with model version
    - Allows joining to actual outcomes for ROI computation
    
    Example:
        store = PredictionStore()
        store.save_prediction(event_id=123, model_version="v1.0", pred_prob=0.65, odds=1.8)
        # After match completes...
        store.record_outcome(event_id=123, actual_outcome=True)
        roi = store.compute_roi(model_version="v1.0")
    """
    db_path: Path = Path("data/predictions.db")
    
    def __post_init__(self):
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL,
                    model_version TEXT NOT NULL,
                    predicted_prob REAL NOT NULL,
                    predicted_outcome INTEGER NOT NULL,
                    odds REAL NOT NULL,
                    stake REAL DEFAULT 0,
                    surface TEXT,
                    tournament TEXT,
                    odds_band TEXT,
                    created_at TEXT NOT NULL,
                    
                    -- Outcome fields
                    actual_outcome INTEGER,
                    pnl REAL,
                    resolved_at TEXT,
                    
                    UNIQUE(event_id, model_version)
                );
                
                CREATE INDEX IF NOT EXISTS idx_predictions_model 
                ON predictions(model_version);
                
                CREATE INDEX IF NOT EXISTS idx_predictions_unresolved 
                ON predictions(actual_outcome) WHERE actual_outcome IS NULL;
                
                CREATE INDEX IF NOT EXISTS idx_predictions_created 
                ON predictions(created_at);
            """)
            conn.commit()
            logger.info(f"Initialized prediction store at {self.db_path}")
        finally:
            conn.close()
    
    def _get_odds_band(self, odds: float) -> str:
        """Categorize odds into bands."""
        if odds < 1.3:
            return "heavy_favorite"
        elif odds < 1.6:
            return "favorite"
        elif odds < 2.0:
            return "slight_favorite"
        elif odds < 2.5:
            return "even"
        elif odds < 3.5:
            return "underdog"
        else:
            return "long_shot"
    
    def save_prediction(
        self,
        event_id: int,
        model_version: str,
        predicted_prob: float,
        odds: float,
        stake: float = 0.0,
        surface: str = None,
        tournament: str = None,
    ) -> bool:
        """
        Save a new prediction.
        
        Args:
            event_id: Unique event identifier
            model_version: Model version used for prediction
            predicted_prob: Predicted probability of player winning
            odds: Decimal odds for the player
            stake: Amount staked (0 if no bet)
            surface: Surface type
            tournament: Tournament name
            
        Returns:
            True if saved successfully
        """
        predicted_outcome = predicted_prob >= 0.5
        odds_band = self._get_odds_band(odds)
        
        conn = self._get_conn()
        try:
            conn.execute("""
                INSERT OR REPLACE INTO predictions 
                (event_id, model_version, predicted_prob, predicted_outcome, 
                 odds, stake, surface, tournament, odds_band, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_id, model_version, predicted_prob, int(predicted_outcome),
                odds, stake, surface, tournament, odds_band,
                datetime.now().isoformat()
            ))
            conn.commit()
            logger.debug(f"Saved prediction for event {event_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return False
        finally:
            conn.close()
    
    def record_outcome(
        self,
        event_id: int,
        actual_outcome: bool,
        model_version: str = None,
    ) -> int:
        """
        Record actual outcome for a prediction.
        
        Args:
            event_id: Event identifier
            actual_outcome: True if player won
            model_version: Optionally filter by model version
            
        Returns:
            Number of predictions updated
        """
        conn = self._get_conn()
        try:
            # Get predictions for this event
            if model_version:
                rows = conn.execute("""
                    SELECT id, odds, stake FROM predictions 
                    WHERE event_id = ? AND model_version = ? AND actual_outcome IS NULL
                """, (event_id, model_version)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT id, odds, stake FROM predictions 
                    WHERE event_id = ? AND actual_outcome IS NULL
                """, (event_id,)).fetchall()
            
            updated = 0
            for row in rows:
                # Calculate PnL
                if row['stake'] > 0:
                    if actual_outcome:
                        pnl = row['stake'] * (row['odds'] - 1)
                    else:
                        pnl = -row['stake']
                else:
                    pnl = 0.0
                
                conn.execute("""
                    UPDATE predictions 
                    SET actual_outcome = ?, pnl = ?, resolved_at = ?
                    WHERE id = ?
                """, (int(actual_outcome), pnl, datetime.now().isoformat(), row['id']))
                updated += 1
            
            conn.commit()
            logger.info(f"Recorded outcome for event {event_id}: {updated} predictions updated")
            return updated
            
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            return 0
        finally:
            conn.close()
    
    def get_pending_outcomes(self, max_age_days: int = 7) -> List[int]:
        """Get event IDs with unresolved outcomes."""
        conn = self._get_conn()
        try:
            cutoff = datetime.now().isoformat()[:10]  # Just date portion
            rows = conn.execute("""
                SELECT DISTINCT event_id FROM predictions 
                WHERE actual_outcome IS NULL 
                AND date(created_at) >= date(?, '-' || ? || ' days')
            """, (cutoff, max_age_days)).fetchall()
            return [row['event_id'] for row in rows]
        finally:
            conn.close()
    
    def compute_roi(
        self,
        model_version: str = None,
        start_date: date = None,
        end_date: date = None,
        surface: str = None,
        odds_band: str = None,
    ) -> Dict[str, float]:
        """
        Compute ROI and statistics for resolved predictions.
        
        Returns:
            Dict with total_bets, total_stake, total_pnl, roi, win_rate
        """
        conn = self._get_conn()
        try:
            query = """
                SELECT 
                    COUNT(*) as total_bets,
                    SUM(stake) as total_stake,
                    SUM(pnl) as total_pnl,
                    SUM(CASE WHEN actual_outcome = predicted_outcome THEN 1 ELSE 0 END) as correct,
                    AVG(predicted_prob) as avg_confidence
                FROM predictions
                WHERE actual_outcome IS NOT NULL
                AND stake > 0
            """
            params = []
            
            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)
            if start_date:
                query += " AND date(created_at) >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date(created_at) <= ?"
                params.append(end_date.isoformat())
            if surface:
                query += " AND surface = ?"
                params.append(surface)
            if odds_band:
                query += " AND odds_band = ?"
                params.append(odds_band)
            
            row = conn.execute(query, params).fetchone()
            
            if not row or row['total_bets'] == 0:
                return {
                    "total_bets": 0,
                    "total_stake": 0.0,
                    "total_pnl": 0.0,
                    "roi": 0.0,
                    "win_rate": 0.0,
                }
            
            total_stake = row['total_stake'] or 0.0
            total_pnl = row['total_pnl'] or 0.0
            
            return {
                "total_bets": row['total_bets'],
                "total_stake": total_stake,
                "total_pnl": total_pnl,
                "roi": (total_pnl / total_stake * 100) if total_stake > 0 else 0.0,
                "win_rate": (row['correct'] / row['total_bets'] * 100) if row['total_bets'] > 0 else 0.0,
                "avg_confidence": row['avg_confidence'] or 0.0,
            }
            
        finally:
            conn.close()
    
    def get_model_versions(self) -> List[str]:
        """Get all model versions with predictions."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT DISTINCT model_version FROM predictions
                ORDER BY model_version DESC
            """).fetchall()
            return [row['model_version'] for row in rows]
        finally:
            conn.close()
    
    def get_daily_stats(
        self,
        model_version: str = None,
        days: int = 30,
    ) -> List[Dict]:
        """Get daily statistics for graphing."""
        conn = self._get_conn()
        try:
            query = """
                SELECT 
                    date(created_at) as date,
                    COUNT(*) as bets,
                    SUM(stake) as stake,
                    SUM(CASE WHEN actual_outcome IS NOT NULL THEN pnl ELSE 0 END) as pnl,
                    SUM(CASE WHEN actual_outcome = predicted_outcome THEN 1 ELSE 0 END) as wins
                FROM predictions
                WHERE date(created_at) >= date('now', '-' || ? || ' days')
            """
            params = [days]
            
            if model_version:
                query += " AND model_version = ?"
                params.append(model_version)
            
            query += " GROUP BY date(created_at) ORDER BY date"
            
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
            
        finally:
            conn.close()


# Global instance
_prediction_store = None


def get_prediction_store(db_path: Path = None) -> PredictionStore:
    """Get global prediction store instance."""
    global _prediction_store
    if _prediction_store is None:
        if db_path:
            _prediction_store = PredictionStore(db_path=db_path)
        else:
            _prediction_store = PredictionStore()
    return _prediction_store
