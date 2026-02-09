"""
🎯 Predictions Page - Upcoming Value Bets
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import polars as pl
from datetime import datetime

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

st.title("🎯 Upcoming Predictions")
st.markdown("---")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    days = st.slider("Days ahead", 1, 14, 3)
    min_odds = st.slider("Min odds", 1.0, 3.0, 1.5, 0.1)
    max_odds = st.slider("Max odds", 1.5, 10.0, 3.0, 0.1)
    min_confidence = st.slider("Min confidence", 0.5, 0.8, 0.55, 0.05)
    
    refresh = st.button("🔄 Refresh Data", type="primary")

# Load predictions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_predictions(days, min_odds, max_odds, min_confidence):
    try:
        from src.pipeline import TennisPipeline
        pipeline = TennisPipeline()
        return pipeline.predict_upcoming(
            days=days,
            min_odds=min_odds,
            max_odds=max_odds,
            min_confidence=min_confidence,
            skip_scrape=True  # Use cached data for speed
        )
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return pl.DataFrame()

if refresh:
    st.cache_data.clear()
    st.rerun()

with st.spinner("Loading predictions..."):
    predictions = load_predictions(days, min_odds, max_odds, min_confidence)

if len(predictions) == 0:
    st.info("No predictions available. Try adjusting filters or run: `python tennis.py predict`")
else:
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bets", len(predictions))
    with col2:
        avg_edge = predictions["edge"].mean() * 100 if "edge" in predictions.columns else 0
        st.metric("Avg Edge", f"{avg_edge:.1f}%")
    with col3:
        avg_odds = predictions["odds_player"].mean() if "odds_player" in predictions.columns else 0
        st.metric("Avg Odds", f"{avg_odds:.2f}")
    with col4:
        avg_conf = predictions["model_prob"].mean() * 100 if "model_prob" in predictions.columns else 0
        st.metric("Avg Confidence", f"{avg_conf:.0f}%")
    
    st.markdown("---")
    
    # Display table
    display_cols = ["player_name", "opponent_name", "model_prob", "odds_player", "edge", "tournament_name"]
    available_cols = [c for c in display_cols if c in predictions.columns]
    
    if available_cols:
        df = predictions.select(available_cols).to_pandas()
        
        # Format columns
        if "model_prob" in df.columns:
            df["model_prob"] = (df["model_prob"] * 100).round(1).astype(str) + "%"
        if "edge" in df.columns:
            df["edge"] = (df["edge"] * 100).round(1).astype(str) + "%"
        if "odds_player" in df.columns:
            df["odds_player"] = df["odds_player"].round(2)
        
        df.columns = ["Bet On", "Opponent", "Confidence", "Odds", "Edge", "Tournament"][:len(df.columns)]
        
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(predictions.to_pandas(), use_container_width=True, hide_index=True)
    
    # Export
    st.download_button(
        "📥 Download CSV",
        predictions.to_pandas().to_csv(index=False),
        "predictions.csv",
        "text/csv"
    )
