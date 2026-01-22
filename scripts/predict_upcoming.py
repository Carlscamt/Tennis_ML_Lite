"""
Daily Predictions Script

Generates predictions and value bets for upcoming matches.
Outputs to console and saves to file for dashboard.

Usage:
    python scripts/predict_upcoming.py
    python scripts/predict_upcoming.py --min-edge 0.05 --min-conf 0.55
"""
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import polars as pl

from config import PROCESSED_DATA_DIR, MODELS_DIR, BETTING
from src.model import Predictor, ModelRegistry
from src.transform import FeatureEngineer
from src.utils import setup_logging

logger = setup_logging()

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = ROOT / "data"
FUTURE_DIR = DATA_DIR / "future"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def load_trained_players() -> set:
    """Load set of player IDs from original training data."""
    try:
        hist_df = pl.read_parquet(PROCESSED_DATA_DIR / "features_dataset.parquet")
        return set(hist_df["player_id"].unique().to_list())
    except:
        logger.warning("Could not load trained players set")
        return set()


def calculate_confidence(
    player_id: int,
    opponent_id: int,
    historical_df: pl.DataFrame,
    trained_players: set
) -> tuple[int, str]:
    """Calculate confidence score (0-100) and tier for a prediction.
    
    Args:
        player_id: Player ID
        opponent_id: Opponent ID
        historical_df: Historical match data
        trained_players: Set of player IDs in training set
    
    Returns:
        (confidence_score, confidence_tier)
    """
    score = 0
    
    # Check if players were in training set (40pts each)
    player_in_training = player_id in trained_players
    opponent_in_training = opponent_id in trained_players
    
    if player_in_training:
        score += 40
    if opponent_in_training:
        score += 40
    
    # Count recent match history (up to 10pts each player)
    try:
        player_matches = len(historical_df.filter(pl.col("player_id") == player_id))
        opponent_matches = len(historical_df.filter(pl.col("player_id") == opponent_id))
        
        # Award points based on match count (5 matches = 1pt, max 10pts)
        score += min(10, player_matches // 5)
        score += min(10, opponent_matches // 5)
    except:
        pass
    
    # Determine tier
    if score >= 75:
        tier = "High"
    elif score >= 50:
        tier = "Medium"
    else:
        tier = "Low"
    
    return (min(100, score), tier)


# =============================================================================
# FEATURE ENGINEERING FOR FUTURE MATCHES
# =============================================================================

def add_features_for_prediction(
    future_df: pl.DataFrame,
    historical_df: pl.DataFrame,
    fe: FeatureEngineer,
) -> pl.DataFrame:
    """
    Add rolling features to future matches using historical data.
    
    For each player in future matches:
    - Get their historical stats from the processed dataset
    - Apply latest values as features
    """
    # Get latest stats per player from historical data
    # This is a simplified approach - take latest row per player
    
    latest_stats = historical_df.group_by("player_id").agg([
        # Rolling win rates
        pl.col("player_win_rate_5").last().alias("player_win_rate_5"),
        pl.col("player_win_rate_10").last().alias("player_win_rate_10") if "player_win_rate_10" in historical_df.columns else pl.lit(None).alias("player_win_rate_10"),
        pl.col("player_win_rate_20").last().alias("player_win_rate_20") if "player_win_rate_20" in historical_df.columns else pl.lit(None).alias("player_win_rate_20"),
        
        # H2H (placeholder - would need matchup-specific lookup)
        pl.col("h2h_wins").last().alias("player_h2h_wins") if "h2h_wins" in historical_df.columns else pl.lit(0).alias("player_h2h_wins"),
        pl.col("h2h_matches").last().alias("player_h2h_matches") if "h2h_matches" in historical_df.columns else pl.lit(0).alias("player_h2h_matches"),
        
        # Days since last match
        pl.col("match_date").last().alias("last_match_date") if "match_date" in historical_df.columns else pl.lit(None).alias("last_match_date"),
        
        # Surface win rate
        pl.col("player_surface_win_rate_10").last().alias("player_surface_win_rate_10") if "player_surface_win_rate_10" in historical_df.columns else pl.lit(None).alias("player_surface_win_rate_10"),
    ])
    
    # Join player stats
    future_df = future_df.join(
        latest_stats.rename({"player_id": "player_id_join"}),
        left_on="player_id",
        right_on="player_id_join",
        how="left"
    )
    
    # Get opponent stats
    opponent_stats = latest_stats.rename({
        "player_id": "opponent_id_join",
        "player_win_rate_5": "opponent_win_rate_5",
        "player_win_rate_10": "opponent_win_rate_10",
        "player_win_rate_20": "opponent_win_rate_20",
        "player_h2h_wins": "opponent_h2h_wins",
        "player_h2h_matches": "opponent_h2h_matches",
        "last_match_date": "opponent_last_match_date",
        "player_surface_win_rate_10": "opponent_surface_win_rate_10",
    })
    
    future_df = future_df.join(
        opponent_stats,
        left_on="opponent_id",
        right_on="opponent_id_join",
        how="left"
    )
    
    # Add odds features
    if "odds_player" in future_df.columns:
        future_df = future_df.with_columns([
            (1 / pl.col("odds_player")).alias("implied_prob_player"),
            (1 / pl.col("odds_opponent")).alias("implied_prob_opponent"),
            (pl.col("odds_opponent") / pl.col("odds_player")).alias("odds_ratio"),
            (pl.col("odds_player") > 2.0).alias("is_underdog"),
        ])
    
    return future_df


# =============================================================================
# PREDICTION
# =============================================================================

def predict_upcoming_matches(
    min_edge: float = 0.05,
    min_confidence: float = 0.55,
    min_odds: float = 1.30,
    max_odds: float = 5.00,
) -> pl.DataFrame:
    """
    Generate predictions for upcoming matches.
    """
    logger.info("="*60)
    logger.info("GENERATING PREDICTIONS")
    logger.info("="*60)
    
    # Load future matches
    future_path = FUTURE_DIR / "upcoming_matches_latest.parquet"
    if not future_path.exists():
        logger.error("No future matches found. Run scrape_future.py first.")
        return pl.DataFrame()
    
    future_df = pl.read_parquet(future_path)
    logger.info(f"Future matches loaded: {len(future_df)}")
    
    # Filter to matches with odds
    future_df = future_df.filter(pl.col("has_odds"))
    logger.info(f"With odds: {len(future_df)}")
    
    # Filter to tournament types where model has coverage
    # Model trained on ATP singles, so exclude UTR, Doubles, ITF
    # Filter to tournament types where model has coverage
    # Model trained on ATP singles, so exclude UTR, Doubles, ITF
    valid_types = ["ATP", "Grand Slam", "ATP/WTA Tour", "Challenger"]
    
    if "tournament_type" in future_df.columns:
        # Filter out explicit WTA, UTR, ITF, Doubles
        future_df = future_df.filter(
            pl.col("tournament_type").is_in(valid_types)
        )
        logger.info(f"After tournament filter ({valid_types}): {len(future_df)}")
    
    # Filter out doubles matches (model trained on singles only)
    if "tournament_name" in future_df.columns:
        future_df = future_df.filter(
            ~pl.col("tournament_name").str.to_lowercase().str.contains("doubles")
        )
        logger.info(f"After doubles filter (singles only): {len(future_df)}")
    
    if len(future_df) == 0:
        logger.warning("No matches matches criteria found.")
        return pl.DataFrame()
    
    # Load historical data for feature engineering
    hist_path = PROCESSED_DATA_DIR / "features_dataset.parquet"
    if hist_path.exists():
        historical_df = pl.read_parquet(hist_path)
        logger.info(f"Historical features loaded: {len(historical_df):,} matches")
    else:
        logger.warning("No historical features found. Predictions will be limited.")
        historical_df = pl.DataFrame()
    
    # Add features
    if len(historical_df) > 0:
        fe = FeatureEngineer()
        future_df = add_features_for_prediction(future_df, historical_df, fe)
    
    # Load model
    registry = ModelRegistry(MODELS_DIR)
    active = registry.get_active_model()
    
    if not active:
        logger.error("No active model. Run training first.")
        return pl.DataFrame()
    
    model_path = Path(active["path"])
    predictor = Predictor(model_path)
    
    # Get predictions
    logger.info("Generating predictions...")
    predictions_df = predictor.predict_with_value(
        future_df.lazy().collect(),
        min_edge=min_edge
    )
    
    # Add confidence scoring
    logger.info("Calculating confidence scores...")
    trained_players = load_trained_players()
    
    confidence_scores = []
    confidence_tiers = []
    
    for row in predictions_df.iter_rows(named=True):
        player_id = row.get("player_id")
        opponent_id = row.get("opponent_id")
        
        score, tier = calculate_confidence(
            player_id,
            opponent_id,
            historical_df if len(historical_df) > 0 else pl.DataFrame(),
            trained_players
        )
        
        confidence_scores.append(score)
        confidence_tiers.append(tier)
    
    # Add confidence columns
    predictions_df = predictions_df.with_columns([
        pl.Series("confidence_score", confidence_scores),
        pl.Series("confidence_tier", confidence_tiers)
    ])
    
    # Filter value bets (now with confidence considerations)
    value_bets = predictions_df.filter(
        (pl.col("edge") >= min_edge) &
        (pl.col("model_prob") >= min_confidence) &
        (pl.col("odds_player") >= min_odds) &
        (pl.col("odds_player") <= max_odds)
    ).sort("edge", descending=True)
    
    logger.info(f"Value bets found: {len(value_bets)}")
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Full predictions
    pred_path = PREDICTIONS_DIR / f"predictions_{timestamp}.parquet"
    predictions_df.write_parquet(pred_path)
    
    # Latest file
    latest_path = PREDICTIONS_DIR / "predictions_latest.parquet"
    predictions_df.write_parquet(latest_path)
    
    # Value bets JSON for easy reading
    if len(value_bets) > 0:
        value_bets_output = []
        for row in value_bets.iter_rows(named=True):
            value_bets_output.append({
                "event_id": row.get("event_id", ""),
                "match_date": row.get("match_date", ""),
                "match_time": row.get("match_time", ""),
                "tournament": row.get("tournament_name", ""),
                "player": row.get("player_name", ""),
                "opponent": row.get("opponent_name", ""),
                "odds": row.get("odds_player", 0),
                "model_prob": round(row.get("model_prob", 0) * 100, 1),
                "implied_prob": round(row.get("implied_prob_player", 0) * 100, 1),
                "edge": round(row.get("edge", 0) * 100, 1),
                "confidence_score": row.get("confidence_score", 0),
                "confidence_tier": row.get("confidence_tier", "Low"),
                "recommendation": "BET" if row.get("edge", 0) > 0.05 else "WATCH"
            })
        
        json_path = PREDICTIONS_DIR / "value_bets_latest.json"
        with open(json_path, "w") as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "criteria": {
                    "min_edge": min_edge,
                    "min_confidence": min_confidence,
                    "min_odds": min_odds,
                    "max_odds": max_odds,
                },
                "value_bets": value_bets_output,
            }, f, indent=2)
        
        logger.info(f"Saved: {json_path}")
    
    return value_bets


def display_value_bets(df: pl.DataFrame):
    """Display value bets in a nice format."""
    if len(df) == 0:
        print("\nNo value bets found for upcoming matches.")
        return
    
    print("\n" + "="*80)
    print("VALUE BET RECOMMENDATIONS")
    print("="*80)
    print()
    
    for i, row in enumerate(df.iter_rows(named=True), 1):
        match_date = row.get("match_date", "")
        match_time = row.get("match_time", "")
        tournament = str(row.get("tournament_name", "Unknown")).encode('ascii', 'replace').decode('ascii')
        player = str(row.get("player_name", "Unknown")).encode('ascii', 'replace').decode('ascii')
        opponent = str(row.get("opponent_name", "Unknown")).encode('ascii', 'replace').decode('ascii')
        odds = row.get("odds_player", 0)
        prob = row.get("model_prob", 0) * 100
        implied = row.get("implied_prob_player", 0) * 100
        edge = row.get("edge", 0) * 100
        
        print(f"[{i}] {match_date} {match_time}")
        print(f"    {tournament}")
        print(f"    {player} vs {opponent}")
        print(f"    Odds: {odds:.2f} | Model: {prob:.1f}% | Implied: {implied:.1f}% | Edge: +{edge:.1f}%")
        print(f"    >>> BET ON: {player}")
        print()
    
    print("="*80)
    print(f"Total Value Bets: {len(df)}")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Predict Upcoming Matches")
    parser.add_argument("--min-edge", type=float, default=BETTING.min_edge,
                       help="Minimum edge to consider")
    parser.add_argument("--min-conf", type=float, default=BETTING.min_confidence,
                       help="Minimum model confidence")
    parser.add_argument("--min-odds", type=float, default=BETTING.min_odds,
                       help="Minimum odds")
    parser.add_argument("--max-odds", type=float, default=BETTING.max_odds,
                       help="Maximum odds")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()
    
    value_bets = predict_upcoming_matches(
        min_edge=args.min_edge,
        min_confidence=args.min_conf,
        min_odds=args.min_odds,
        max_odds=args.max_odds,
    )
    
    if not args.quiet:
        display_value_bets(value_bets)
    
    return 0 if len(value_bets) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
