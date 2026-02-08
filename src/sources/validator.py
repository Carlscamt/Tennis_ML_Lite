"""
Cross-source validation and anomaly detection.

Compares data between sources to detect inconsistencies and
alert on significant divergences that may indicate data quality issues.
"""
import logging
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import polars as pl

from .canonical import CanonicalMatch, DataSource, canonical_to_dataframe

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Detected anomaly between sources."""
    anomaly_type: str
    severity: str  # "low", "medium", "high"
    description: str
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    source1_value: Optional[float] = None
    source2_value: Optional[float] = None
    divergence: Optional[float] = None


@dataclass
class ValidationReport:
    """Summary of cross-source validation."""
    timestamp: str
    matches_compared: int
    anomalies: List[Anomaly] = field(default_factory=list)
    win_rate_divergences: Dict[str, float] = field(default_factory=dict)
    overall_status: str = "unknown"  # "healthy", "warning", "alert"
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "matches_compared": self.matches_compared,
            "anomaly_count": len(self.anomalies),
            "overall_status": self.overall_status,
            "win_rate_divergences": self.win_rate_divergences,
        }


@dataclass
class CrossSourceValidator:
    """
    Validates data consistency between multiple sources.
    
    Compares key aggregates (win rates, match counts) between sources
    and alerts on significant divergences.
    
    Example:
        validator = CrossSourceValidator()
        validator.add_matches(sofascore_matches, DataSource.SOFASCORE)
        validator.add_matches(sackmann_matches, DataSource.SACKMANN)
        
        report = validator.validate()
        if report.overall_status == "alert":
            notify_admin(report)
    """
    divergence_threshold: float = 0.10  # 10% divergence triggers alert
    min_matches_for_comparison: int = 10
    
    _matches: Dict[DataSource, List[CanonicalMatch]] = field(default_factory=dict)
    
    def __post_init__(self):
        self._matches = defaultdict(list)
    
    def add_matches(self, matches: List[CanonicalMatch], source: DataSource = None):
        """Add matches for validation."""
        for match in matches:
            src = source or match.source
            self._matches[src].append(match)
    
    def clear(self):
        """Clear all matches."""
        self._matches.clear()
    
    def compare_win_rates(
        self,
        player_id: int = None,
        player_name: str = None,
        start_date: date = None,
        end_date: date = None,
    ) -> Dict[str, Dict]:
        """
        Compare win rates for a player across sources.
        
        Args:
            player_id: Player ID to compare (optional)
            player_name: Player name to search (optional)
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            Dict with win rates by source
        """
        results = {}
        
        for source, matches in self._matches.items():
            # Filter by player
            player_matches = []
            for m in matches:
                if player_id and (m.winner_id == player_id or m.loser_id == player_id):
                    player_matches.append(m)
                elif player_name and (
                    player_name.lower() in m.winner_name.lower() or
                    player_name.lower() in m.loser_name.lower()
                ):
                    player_matches.append(m)
            
            # Filter by date
            if start_date:
                player_matches = [m for m in player_matches if m.match_date and m.match_date >= start_date]
            if end_date:
                player_matches = [m for m in player_matches if m.match_date and m.match_date <= end_date]
            
            if not player_matches:
                continue
            
            # Calculate win rate
            wins = sum(1 for m in player_matches if m.winner_id == player_id or 
                      (player_name and player_name.lower() in m.winner_name.lower()))
            total = len(player_matches)
            
            results[source.value] = {
                "wins": wins,
                "losses": total - wins,
                "total": total,
                "win_rate": wins / total if total > 0 else 0.0,
            }
        
        return results
    
    def detect_anomalies(self) -> List[Anomaly]:
        """
        Detect anomalies between sources.
        
        Currently checks:
        - Win rate divergence for top players
        - Match count discrepancies
        
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if len(self._matches) < 2:
            return anomalies
        
        # Get all unique player IDs
        all_players = set()
        for matches in self._matches.values():
            for m in matches:
                all_players.add((m.winner_id, m.winner_name))
                all_players.add((m.loser_id, m.loser_name))
        
        # Compare win rates for players with enough matches
        for player_id, player_name in all_players:
            if not player_id:
                continue
            
            rates = self.compare_win_rates(player_id=player_id)
            
            if len(rates) < 2:
                continue
            
            # Check if player has enough matches in both sources
            if all(r["total"] >= self.min_matches_for_comparison for r in rates.values()):
                win_rates = [r["win_rate"] for r in rates.values()]
                divergence = abs(max(win_rates) - min(win_rates))
                
                if divergence > self.divergence_threshold:
                    anomalies.append(Anomaly(
                        anomaly_type="win_rate_divergence",
                        severity="high" if divergence > 0.20 else "medium",
                        description=f"Win rate divergence of {divergence:.1%} for player {player_name}",
                        player_id=player_id,
                        player_name=player_name,
                        source1_value=win_rates[0],
                        source2_value=win_rates[1],
                        divergence=divergence,
                    ))
        
        return anomalies
    
    def validate(self) -> ValidationReport:
        """
        Run full validation and generate report.
        
        Returns:
            ValidationReport with findings
        """
        from datetime import datetime
        
        anomalies = self.detect_anomalies()
        
        # Count total matches compared
        total_matches = sum(len(m) for m in self._matches.values())
        
        # Determine overall status
        high_severity = sum(1 for a in anomalies if a.severity == "high")
        medium_severity = sum(1 for a in anomalies if a.severity == "medium")
        
        if high_severity > 0:
            status = "alert"
        elif medium_severity > 2:
            status = "warning"
        else:
            status = "healthy"
        
        # Win rate divergences summary
        divergences = {}
        for a in anomalies:
            if a.anomaly_type == "win_rate_divergence" and a.player_name:
                divergences[a.player_name] = a.divergence
        
        return ValidationReport(
            timestamp=datetime.now().isoformat(),
            matches_compared=total_matches,
            anomalies=anomalies,
            win_rate_divergences=divergences,
            overall_status=status,
        )
    
    def generate_report_text(self) -> str:
        """Generate human-readable validation report."""
        report = self.validate()
        
        lines = [
            "=" * 50,
            "CROSS-SOURCE VALIDATION REPORT",
            "=" * 50,
            f"Timestamp: {report.timestamp}",
            f"Status: {report.overall_status.upper()}",
            f"Matches compared: {report.matches_compared}",
            f"Anomalies found: {len(report.anomalies)}",
            "",
        ]
        
        if report.anomalies:
            lines.append("ANOMALIES:")
            lines.append("-" * 30)
            for a in report.anomalies[:10]:  # Limit to 10
                lines.append(f"  [{a.severity.upper()}] {a.description}")
        else:
            lines.append("No anomalies detected.")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


def quick_validate(
    sofascore_df: pl.DataFrame,
    sackmann_df: pl.DataFrame,
    threshold: float = 0.10,
) -> ValidationReport:
    """
    Quick validation between two DataFrames.
    
    Args:
        sofascore_df: SofaScore data
        sackmann_df: Sackmann data
        threshold: Divergence threshold for alerts
        
    Returns:
        ValidationReport
    """
    from .canonical import to_canonical, DataSource
    
    validator = CrossSourceValidator(divergence_threshold=threshold)
    
    sofascore_matches = to_canonical(sofascore_df, DataSource.SOFASCORE)
    sackmann_matches = to_canonical(sackmann_df, DataSource.SACKMANN)
    
    validator.add_matches(sofascore_matches)
    validator.add_matches(sackmann_matches)
    
    return validator.validate()
