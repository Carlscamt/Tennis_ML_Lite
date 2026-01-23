"""
Data quality validation utilities.
"""
import polars as pl
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    details: Optional[dict] = None


class DataValidator:
    """
    Validate data quality for ML training.
    """
    
    required_columns = [
        "event_id",
        "player_id",
        "opponent_id",
        "player_won",
        "start_timestamp",
    ]
    
    def __init__(self, min_odds_coverage: float = 0.7):
        """
        Args:
            min_odds_coverage: Minimum fraction of matches with odds data
        """
        self.min_odds_coverage = min_odds_coverage
        self.results: List[ValidationResult] = []
    
    def validate_all(self, df: pl.LazyFrame) -> bool:
        """
        Run all validation checks.
        
        Args:
            df: LazyFrame to validate
            
        Returns:
            True if all checks pass
        """
        self.results = []
        
        self.results.append(self._check_required_columns(df))
        self.results.append(self._check_no_duplicates(df))
        self.results.append(self._check_valid_targets(df))
        self.results.append(self._check_odds_coverage(df))
        self.results.append(self._check_date_range(df))
        
        all_passed = all(r.passed for r in self.results)
        
        # Log results
        for result in self.results:
            status = "✓" if result.passed else "✗"
            logger.info(f"{status} {result.message}")
        
        return all_passed
    
    def _check_required_columns(self, df: pl.LazyFrame) -> ValidationResult:
        """Check all required columns exist."""
        schema = df.collect_schema().names()
        missing = [col for col in self.required_columns if col not in schema]
        
        if missing:
            return ValidationResult(
                passed=False,
                message=f"Missing required columns: {missing}",
                details={"missing": missing}
            )
        
        return ValidationResult(
            passed=True,
            message="All required columns present"
        )
    
    def _check_no_duplicates(self, df: pl.LazyFrame) -> ValidationResult:
        """Check for duplicate event_id + player_id combinations."""
        dups = df.group_by(["event_id", "player_id"]).agg(
            pl.len().alias("count")
        ).filter(pl.col("count") > 1).collect()
        
        if len(dups) > 0:
            return ValidationResult(
                passed=False,
                message=f"Found {len(dups)} duplicate event-player combinations",
                details={"duplicate_count": len(dups)}
            )
        
        return ValidationResult(
            passed=True,
            message="No duplicate matches found"
        )
    
    def _check_valid_targets(self, df: pl.LazyFrame) -> ValidationResult:
        """Check target column has valid values."""
        stats = df.select([
            pl.col("player_won").is_null().sum().alias("null_count"),
            pl.col("player_won").sum().alias("wins"),
            pl.len().alias("total")
        ]).collect().to_dicts()[0]
        
        if stats["null_count"] > 0:
            return ValidationResult(
                passed=False,
                message=f"Found {stats['null_count']} null target values",
                details=stats
            )
        
        win_rate = stats["wins"] / stats["total"]
        if win_rate < 0.3 or win_rate > 0.7:
            return ValidationResult(
                passed=False,
                message=f"Suspicious win rate: {win_rate:.1%} (expected ~50%)",
                details={"win_rate": win_rate}
            )
        
        return ValidationResult(
            passed=True,
            message=f"Valid targets (win rate: {win_rate:.1%})"
        )
    
    def _check_odds_coverage(self, df: pl.LazyFrame) -> ValidationResult:
        """Check sufficient odds data coverage."""
        schema = df.collect_schema().names()
        
        if "odds_player" not in schema:
            return ValidationResult(
                passed=False,
                message="No odds_player column found",
            )
        
        stats = df.select([
            pl.col("odds_player").is_not_null().sum().alias("with_odds"),
            pl.len().alias("total")
        ]).collect().to_dicts()[0]
        
        coverage = stats["with_odds"] / stats["total"]
        
        if coverage < self.min_odds_coverage:
            return ValidationResult(
                passed=False,
                message=f"Odds coverage too low: {coverage:.1%} (need {self.min_odds_coverage:.1%})",
                details={"coverage": coverage}
            )
        
        return ValidationResult(
            passed=True,
            message=f"Odds coverage: {coverage:.1%}"
        )
    
    def _check_date_range(self, df: pl.LazyFrame) -> ValidationResult:
        """Check data spans reasonable date range."""
        stats = df.select([
            pl.from_epoch("start_timestamp").min().alias("earliest"),
            pl.from_epoch("start_timestamp").max().alias("latest"),
        ]).collect().to_dicts()[0]
        
        return ValidationResult(
            passed=True,
            message=f"Date range: {stats['earliest'].date()} to {stats['latest'].date()}",
            details=stats
        )
    
    def get_report(self) -> str:
        """Generate validation report."""
        lines = ["Data Validation Report", "=" * 40]
        
        for result in self.results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"[{status}] {result.message}")
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        lines.append("=" * 40)
        lines.append(f"Result: {passed}/{total} checks passed")
        
        return "\n".join(lines)
