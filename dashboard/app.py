"""
Tennis Betting Dashboard

Streamlit app for viewing predictions and tracking bets.

Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
import json
from datetime import datetime, date, timedelta
import time

# Add project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import polars as pl

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Tennis Betting AI",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=300)
def load_historical_data():
    """Load historical match data."""
    # Try to load the most recent processed data
    files = sorted(PROCESSED_DIR.glob("features_dataset*.parquet"))
    if files:
        return pl.read_parquet(files[-1])
    return None


@st.cache_data(ttl=300)
def load_raw_matches():
    """Load raw match data for player stats."""
    from pathlib import Path
    raw_dir = DATA_DIR / "raw"
    files = sorted(raw_dir.glob("atp_matches_*.parquet"))
    if files:
        return pl.read_parquet(files[-1])
    return None

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR = ROOT / "data"
FUTURE_DIR = DATA_DIR / "future"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
BACKTEST_DIR = DATA_DIR / "backtest"

# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_value_bets():
    """Load latest value bets."""
    path = PREDICTIONS_DIR / "value_bets_latest.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_upcoming_matches():
    """Load upcoming matches."""
    path = FUTURE_DIR / "upcoming_matches_latest.parquet"
    if path.exists():
        return pl.read_parquet(path)
    return None


@st.cache_data(ttl=300)
def load_model_info():
    """Load model registry info."""
    registry_path = MODELS_DIR / "registry.json"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            return json.load(f)
    return None


# =============================================================================
# BET LOG STORAGE
# =============================================================================

BET_LOG_PATH = DATA_DIR / "bet_log.json"

def load_bet_log():
    """Load bet log from JSON file."""
    if BET_LOG_PATH.exists():
        with open(BET_LOG_PATH, "r") as f:
            return json.load(f)
    return {"bets": []}

def save_bet_log(bet_log):
    """Save bet log to JSON file."""
    BET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BET_LOG_PATH, "w") as f:
        json.dump(bet_log, f, indent=2, default=str)

def add_bet_to_log(bet_data: dict, stake: float = 10.0):
    """Add a bet to the log. Returns bet_id or None if duplicate."""
    bet_log = load_bet_log()
    
    # Check for duplicate event_id (prevent betting on same match multiple times)
    event_id = bet_data.get("event_id", "")
    if event_id:
        for existing in bet_log["bets"]:
            if existing.get("event_id") == event_id:
                return None  # Duplicate - already bet on this event
    
    new_bet = {
        "id": len(bet_log["bets"]) + 1,
        "player": bet_data.get("player", ""),
        "opponent": bet_data.get("opponent", ""),
        "tournament": bet_data.get("tournament", ""),
        "match_date": bet_data.get("match_date", ""),
        "match_time": bet_data.get("match_time", ""),
        "odds": bet_data.get("odds", 0),
        "stake": stake,
        "model_prob": bet_data.get("model_prob", 0),
        "edge": bet_data.get("edge", 0),
        "outcome": "Pending",
        "profit": 0,
        "placed_at": datetime.now().isoformat(),
        "event_id": event_id
    }
    
    bet_log["bets"].append(new_bet)
    save_bet_log(bet_log)
    return new_bet["id"]

def update_bet_outcome(bet_id: int, outcome: str):
    """Update bet outcome (Won/Lost)."""
    bet_log = load_bet_log()
    
    for bet in bet_log["bets"]:
        if bet["id"] == bet_id:
            bet["outcome"] = outcome
            if outcome == "Won":
                bet["profit"] = bet["stake"] * (bet["odds"] - 1)
            elif outcome == "Lost":
                bet["profit"] = -bet["stake"]
            break
    
    save_bet_log(bet_log)

def get_bet_stats():
    """Calculate betting statistics."""
    bet_log = load_bet_log()
    bets = bet_log.get("bets", [])
    
    if not bets:
        return None
    
    total = len(bets)
    pending = len([b for b in bets if b["outcome"] == "Pending"])
    won = len([b for b in bets if b["outcome"] == "Won"])
    lost = len([b for b in bets if b["outcome"] == "Lost"])
    
    settled = won + lost
    win_rate = won / settled if settled > 0 else 0
    
    total_staked = sum(b["stake"] for b in bets if b["outcome"] != "Pending")
    total_profit = sum(b["profit"] for b in bets if b["outcome"] != "Pending")
    roi = total_profit / total_staked if total_staked > 0 else 0
    
    return {
        "total": total,
        "pending": pending,
        "won": won,
        "lost": lost,
        "win_rate": win_rate,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi
    }


def auto_check_outcomes():
    """
    Automatically check pending bet outcomes using SofaScore API.
    OPTIMIZED with parallel processing.
    Returns count of updated bets.
    """
    import httpx
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    bet_log = load_bet_log()
    pending_bets = [b for b in bet_log.get("bets", []) if b.get("outcome") == "Pending"]
    
    if not pending_bets:
        return 0
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json"
    }
    
    def check_single_bet(bet):
        """Check outcome for a single bet. Returns (bet_id, outcome, event_id) or None."""
        event_id = bet.get("event_id")
        bet_id = bet.get("id")
        player_name = bet.get("player", "")
        
        # If no event_id, try to find by searching scheduled events
        if not event_id:
            match_date = bet.get("match_date", "")
            if match_date and player_name:
                try:
                    search_url = f"https://www.sofascore.com/api/v1/sport/tennis/scheduled-events/{match_date}"
                    response = httpx.get(search_url, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        for event in response.json().get("events", []):
                            home_name = event.get("homeTeam", {}).get("name", "").lower()
                            away_name = event.get("awayTeam", {}).get("name", "").lower()
                            
                            if player_name.lower() in home_name or player_name.lower() in away_name:
                                event_id = str(event.get("id"))
                                break
                except:
                    pass
        
        if not event_id:
            return None
        
        try:
            url = f"https://www.sofascore.com/api/v1/event/{event_id}"
            response = httpx.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return None
            
            event = response.json().get("event", {})
            status = event.get("status", {})
            
            # Not finished yet
            if status.get("type") != "finished":
                return None
            
            winner_code = event.get("winnerCode")
            home_team = event.get("homeTeam", {}).get("name", "")
            away_team = event.get("awayTeam", {}).get("name", "")
            
            # Determine outcome
            if winner_code == 1 and player_name.lower() in home_team.lower():
                return (bet_id, "Won", event_id)
            elif winner_code == 2 and player_name.lower() in away_team.lower():
                return (bet_id, "Won", event_id)
            elif winner_code in [1, 2]:
                return (bet_id, "Lost", event_id)
            
        except:
            pass
        
        return None
    
    # Process all bets in parallel
    updated_count = 0
    results = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(check_single_bet, bet): bet for bet in pending_bets}
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    # Apply updates
    for bet_id, outcome, event_id in results:
        update_bet_outcome(bet_id, outcome)
        # Update event_id if it was found during search
        for bet in bet_log["bets"]:
            if bet["id"] == bet_id and not bet.get("event_id"):
                bet["event_id"] = event_id
        updated_count += 1
    
    # Save any event_id updates
    save_bet_log(bet_log)
    
    return updated_count


@st.cache_data(ttl=3600)
def load_historical_stats():
    """Load historical performance stats."""
    path = PROCESSED_DIR / "features_dataset.parquet"
    if path.exists():
        df = pl.read_parquet(path)
        return {
            "total_matches": len(df),
            "date_range": f"{df['match_date'].min()} to {df['match_date'].max()}",
            "players": df["player_id"].n_unique(),
        }
    return None


# =============================================================================
# CUSTOM CSS - Enhanced for better contrast
# =============================================================================

st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #1a1f2c;
        --bg-card: #262d3d;
        --text-primary: #fafafa;
        --text-secondary: #b8c5d6;
        --accent-green: #00d97e;
        --accent-blue: #4da6ff;
        --accent-red: #ff6b6b;
    }
    
    /* Metric cards - ensure white text */
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b8c5d6 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #00d97e !important;
    }
    
    /* Metric container background */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        border: 1px solid #374151 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Regular text */
    p, span, label {
        color: #e5e7eb !important;
    }
    
    /* Captions */
    .stCaption {
        color: #9ca3af !important;
    }
    
    /* Expander */
    [data-testid="stExpander"] {
        background: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    
    /* Tables */
    .stDataFrame {
        background: #1f2937 !important;
    }
    
    [data-testid="stDataFrameContainer"] th {
        background: #374151 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stDataFrameContainer"] td {
        color: #e5e7eb !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%) !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: #1e3a5f !important;
        color: #e5e7eb !important;
        border: 1px solid #3b82f6 !important;
    }
    
    /* Radio buttons */
    [data-testid="stRadio"] label {
        color: #ffffff !important;
    }
    
    /* Selectbox */
    [data-testid="stSelectbox"] label {
        color: #e5e7eb !important;
    }
    
    /* Form inputs */
    .stTextInput input, .stNumberInput input {
        background: #374151 !important;
        color: #ffffff !important;
        border: 1px solid #4b5563 !important;
    }
    
    /* Value bet cards */
    .bet-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00d97e;
    }
    
    /* Positive/Negative indicators */
    .positive { color: #00d97e !important; }
    .negative { color: #ff6b6b !important; }
    
    /* Markdown tables */
    table {
        background: #1f2937 !important;
    }
    
    th {
        background: #374151 !important;
        color: #ffffff !important;
    }
    
    td {
        color: #e5e7eb !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/000000/tennis.png", width=80)
    st.title("Tennis Betting AI")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["🎯 Value Bets", "📊 Upcoming Matches", "👤 Player History", "💰 Betting Log", "📈 Model Stats"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Cache", use_container_width=True, help="Clear cached data"):
        st.cache_data.clear()
        st.rerun()
    
    # Fetch new matches button
    if st.button("📡 Fetch New Matches", use_container_width=True, help="Fetch matches & generate predictions"):
        progress = st.progress(0, text="Starting...")
        
        try:
            # Step 1: Scrape future matches (direct import, no subprocess)
            progress.progress(15, text="📡 Fetching upcoming matches...")
            
            from scripts.scrape_future import scrape_future_matches, get_active_player_ids
            
            future_df = scrape_future_matches(days_ahead=3)
            
            if future_df is None or len(future_df) == 0:
                st.warning("No matches found. API may be unavailable.")
                progress.empty()
            else:
                st.caption(f"Found {len(future_df)} matches")
                
                # Step 2: Get active player IDs (skip full player update for speed)
                progress.progress(40, text="👤 Identifying active players...")
                active_ids = get_active_player_ids(future_df)
                st.caption(f"Found {len(active_ids)} active players")
                
                # Step 3: Generate predictions (direct import)
                progress.progress(60, text="🎯 Generating predictions...")
                
                try:
                    from scripts.predict_upcoming import main as predict_main
                    predict_main()
                except Exception as pred_err:
                    st.warning(f"Predictions: {pred_err}")
                
                progress.progress(100, text="✅ Complete!")
                st.cache_data.clear()
                st.success(f"✅ Fetched {len(future_df)} matches & generated predictions!")
                time.sleep(1)
                st.rerun()
                
        except Exception as e:
            progress.empty()
            st.error(f"Error: {e}")
            import traceback
            with st.expander("Details"):
                st.code(traceback.format_exc())
    
    # Update Player History button
    if st.button("📊 Update Player History", use_container_width=True, help="Fetch match history & rebuild dataset"):
        progress = st.progress(0, text="Starting player update...")
        
        try:
            # Step 1: Load upcoming matches to get player IDs
            progress.progress(5, text="📋 Loading upcoming matches...")
            
            from scripts.scrape_future import get_active_player_ids
            import polars as pl
            
            future_path = DATA_DIR / "future" / "upcoming_matches_latest.parquet"
            if not future_path.exists():
                st.warning("No upcoming matches found. Fetch matches first.")
                progress.empty()
            else:
                future_df = pl.read_parquet(future_path)
                active_ids = get_active_player_ids(future_df)
                st.caption(f"Found {len(active_ids)} players to update")
                
                # Step 2: Fetch new match data
                progress.progress(15, text=f"📡 Fetching history for {len(active_ids)} players...")
                
                from scripts.update_active_players import update_player_data, get_latest_data
                
                existing_df = get_latest_data()
                updated_df = update_player_data(
                    player_ids=list(active_ids),
                    existing_df=existing_df,
                    max_pages=3,
                    parallel_workers=3,
                    smart_update=True
                )
                
                # Step 3: Save raw data
                progress.progress(50, text="💾 Saving raw data...")
                
                if updated_df is not None and len(updated_df) > 0:
                    RAW_DIR = DATA_DIR / "raw"
                    RAW_DIR.mkdir(parents=True, exist_ok=True)
                    output_path = RAW_DIR / f"atp_matches_updated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                    updated_df.write_parquet(output_path)
                    st.caption(f"Saved {len(updated_df):,} matches to raw data")
                    
                    # Step 4: Run data pipeline to combine all data
                    progress.progress(60, text="⚙️ Running data pipeline (combining datasets)...")
                    
                    from scripts.run_pipeline import run_data_pipeline
                    from config import RAW_DATA_DIR, PROCESSED_DATA_DIR
                    
                    run_data_pipeline(RAW_DATA_DIR, PROCESSED_DATA_DIR)
                    
                    # Step 5: Regenerate predictions with new data
                    progress.progress(85, text="🎯 Regenerating predictions...")
                    
                    try:
                        from scripts.predict_upcoming import main as predict_main
                        predict_main()
                    except Exception as pred_err:
                        st.warning(f"Prediction update: {pred_err}")
                    
                    progress.progress(100, text="✅ Complete!")
                    st.cache_data.clear()
                    st.success(f"✅ Updated {len(active_ids)} players, rebuilt dataset, regenerated predictions!")
                    time.sleep(1)
                    st.rerun()
                else:
                    progress.progress(100, text="✅ No new data")
                    st.info("All players already up to date!")
                    
        except Exception as e:
            progress.empty()
            st.error(f"Error: {e}")
            import traceback
            with st.expander("Details"):
                st.code(traceback.format_exc())
    
    # Model info
    model_info = load_model_info()
    if model_info and model_info.get("active_model"):
        st.markdown("### Active Model")
        active = model_info["active_model"]
        st.caption(f"Version: {active}")
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# =============================================================================
# VALUE BETS PAGE
# =============================================================================

if page == "🎯 Value Bets":
    st.title("🎯 Today's Value Bets")
    
    value_bets_data = load_value_bets()
    
    if value_bets_data and value_bets_data.get("value_bets"):
        bets = value_bets_data["value_bets"]
        criteria = value_bets_data.get("criteria", {})
        generated_at = value_bets_data.get("generated_at", "")
        
        # Parse generated time
        if generated_at:
            try:
                gen_time = datetime.fromisoformat(generated_at)
                st.caption(f"🕐 Generated: {gen_time.strftime('%Y-%m-%d %H:%M')}")
            except:
                pass
        
        # =================================================================
        # FILTER OUT PAST/FINISHED MATCHES
        # =================================================================
        today = date.today().isoformat()
        original_count = len(bets)
        
        # Filter only future/today matches
        active_bets = []
        for bet in bets:
            match_date = bet.get("match_date", "")
            
            # Skip if match date is in the past
            if match_date and match_date < today:
                continue
            
            # For today's matches, check if time has passed (optional deeper check)
            if match_date == today:
                match_time = bet.get("match_time", "")
                if match_time:
                    try:
                        # Parse match time and compare with current time
                        from datetime import datetime as dt
                        match_dt = dt.strptime(f"{match_date} {match_time}", "%Y-%m-%d %H:%M")
                        # Keep if match hasn't started (add 3 hour buffer for ongoing matches)
                        if match_dt < dt.now() - timedelta(hours=3):
                            continue
                    except:
                        pass  # Keep if can't parse
            
            active_bets.append(bet)
        
        bets = active_bets
        
        # Show filter info if bets were removed
        if len(bets) < original_count:
            st.info(f"📅 Filtered out {original_count - len(bets)} past/finished matches. Showing {len(bets)} active bets.")
        
        # =================================================================
        # ENHANCED SUMMARY SECTION
        # =================================================================
        
        st.markdown("### 📊 Summary")
        
        # Calculate advanced stats
        avg_edge = sum(b.get("edge", 0) for b in bets) / len(bets) if bets else 0
        avg_odds = sum(b.get("odds", 0) for b in bets) / len(bets) if bets else 0
        avg_conf = sum(b.get("model_prob", 0) for b in bets) / len(bets) if bets else 0
        max_edge = max(b.get("edge", 0) for b in bets) if bets else 0
        min_odds = min(b.get("odds", 0) for b in bets) if bets else 0
        max_odds = max(b.get("odds", 0) for b in bets) if bets else 0
        
        # Count by date
        dates = {}
        tournaments = {}
        for b in bets:
            d = b.get("match_date", "Unknown")
            t = b.get("tournament", "Unknown")
            dates[d] = dates.get(d, 0) + 1
            tournaments[t] = tournaments.get(t, 0) + 1
        
        # Top row metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Bets", len(bets))
        with col2:
            st.metric("Avg Edge", f"+{avg_edge:.1f}%")
        with col3:
            st.metric("Best Edge", f"+{max_edge:.1f}%")
        with col4:
            st.metric("Avg Odds", f"{avg_odds:.2f}")
        with col5:
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        # Second row - distribution info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📅 By Date:**")
            for d, count in sorted(dates.items()):
                st.caption(f"  {d}: {count} bets")
        with col2:
            st.markdown("**🏆 Top Tournaments:**")
            for t, count in sorted(tournaments.items(), key=lambda x: -x[1])[:5]:
                st.caption(f"  {t}: {count}")
        with col3:
            st.markdown("**📈 Odds Range:**")
            st.caption(f"  Min: {min_odds:.2f}")
            st.caption(f"  Max: {max_odds:.2f}")
            st.caption(f"  Spread: {max_odds - min_odds:.2f}")
        
        st.markdown("---")
        
        # =================================================================
        # FILTERS & SORTING
        # =================================================================
        
        st.markdown("### 🔍 Filters")
        
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            sort_options = ["Edge (High→Low)", "Edge (Low→High)", "Odds (High→Low)", 
                          "Odds (Low→High)", "Confidence (High→Low)", "Date/Time"]
            sort_by = st.selectbox("Sort By", sort_options)
        
        with filter_col2:
            date_options = ["All Dates"] + sorted(list(dates.keys()))
            filter_date = st.selectbox("Date", date_options)
        
        with filter_col3:
            tournament_options = ["All Tournaments"] + sorted(list(tournaments.keys()))
            filter_tournament = st.selectbox("Tournament", tournament_options)
        
        with filter_col4:
            edge_min = st.slider("Min Edge %", 0, 50, 0)
        
        # Apply filters
        filtered_bets = bets.copy()
        
        if filter_date != "All Dates":
            filtered_bets = [b for b in filtered_bets if b.get("match_date") == filter_date]
        
        if filter_tournament != "All Tournaments":
            filtered_bets = [b for b in filtered_bets if b.get("tournament") == filter_tournament]
        
        if edge_min > 0:
            filtered_bets = [b for b in filtered_bets if b.get("edge", 0) >= edge_min]
        
        # Apply sorting
        if sort_by == "Edge (High→Low)":
            filtered_bets.sort(key=lambda x: -x.get("edge", 0))
        elif sort_by == "Edge (Low→High)":
            filtered_bets.sort(key=lambda x: x.get("edge", 0))
        elif sort_by == "Odds (High→Low)":
            filtered_bets.sort(key=lambda x: -x.get("odds", 0))
        elif sort_by == "Odds (Low→High)":
            filtered_bets.sort(key=lambda x: x.get("odds", 0))
        elif sort_by == "Confidence (High→Low)":
            filtered_bets.sort(key=lambda x: -x.get("model_prob", 0))
        elif sort_by == "Date/Time":
            filtered_bets.sort(key=lambda x: (x.get("match_date", ""), x.get("match_time", "")))
        
        st.markdown(f"**Showing {len(filtered_bets)} of {len(bets)} bets**")
        
        st.markdown("---")
        
        # =================================================================
        # PLACE ALL BETS BUTTON
        # =================================================================
        
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                bulk_stake = st.number_input(
                    "Default Stake for All Bets ($)", 
                    min_value=1.0, value=10.0, step=1.0,
                    key="bulk_stake"
                )
            
            with col2:
                if st.button("💰 Place All Bets", use_container_width=True, type="primary"):
                    placed = 0
                    skipped = 0
                    for bet in filtered_bets:
                        result = add_bet_to_log(bet, stake=bulk_stake)
                        if result:
                            placed += 1
                        else:
                            skipped += 1
                    
                    if placed > 0:
                        st.success(f"✅ Placed {placed} bets!")
                    if skipped > 0:
                        st.info(f"⏭️ Skipped {skipped} (already placed)")
                    st.cache_data.clear()
            
            with col3:
                st.metric("Est. Total", f"${bulk_stake * len(filtered_bets):.2f}")
        
        st.markdown("---")
        
        # =================================================================
        # CRITERIA EXPANDER
        # =================================================================
        
        with st.expander("🔧 Current Criteria"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Min Edge", f"{criteria.get('min_edge', 0.05)*100:.0f}%")
            c2.metric("Min Confidence", f"{criteria.get('min_confidence', 0.55)*100:.0f}%")
            c3.metric("Min Odds", f"{criteria.get('min_odds', 1.30):.2f}")
            c4.metric("Max Odds", f"{criteria.get('max_odds', 5.00):.2f}")
        
        st.markdown("---")
        
        # =================================================================
        # BET CARDS
        # =================================================================
        
        for i, bet in enumerate(filtered_bets, 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    # Add edge indicator color
                    edge = bet.get("edge", 0)
                    if edge >= 30:
                        edge_color = "🔥"
                    elif edge >= 20:
                        edge_color = "⭐"
                    else:
                        edge_color = "✓"
                    
                    st.markdown(f"### {edge_color} {i}. {bet.get('player', 'Unknown')}")
                    st.caption(f"vs {bet.get('opponent', 'Unknown')}")
                    st.caption(f"📅 {bet.get('match_date', '')} {bet.get('match_time', '')} | 🏆 {bet.get('tournament', '')}")
                
                with col2:
                    st.metric("Odds", f"{bet.get('odds', 0):.2f}")
                    st.metric("Edge", f"+{bet.get('edge', 0):.1f}%")
                
                with col3:
                    st.metric("Model", f"{bet.get('model_prob', 0):.1f}%")
                    st.metric("Implied", f"{bet.get('implied_prob', 0):.1f}%")
                    
                    # One-click save bet button
                    bet_key = f"save_bet_{i}_{bet.get('event_id', i)}"
                    stake = st.number_input("Stake $", min_value=1.0, value=10.0, step=1.0, key=f"stake_{bet_key}")
                    
                    if st.button("💾 Save Bet", key=bet_key, use_container_width=True):
                        bet_id = add_bet_to_log(bet, stake=stake)
                        if bet_id:
                            st.success(f"✅ Bet #{bet_id} saved!")
                        else:
                            st.warning("⚠️ Already bet on this match!")
                        st.cache_data.clear()
                
                st.markdown("---")
    else:
        st.info("No value bets found. Run the prediction script first:")
        st.code("python scripts/predict_upcoming.py")
        
        # Show how to generate predictions
        st.markdown("### Quick Start")
        st.markdown("""
        1. **Fetch upcoming matches**: `python scripts/scrape_future.py --days 7`
        2. **Generate predictions**: `python scripts/predict_upcoming.py`
        3. **Refresh this page**
        """)


# =============================================================================
# UPCOMING MATCHES PAGE
# =============================================================================

elif page == "📊 Upcoming Matches":
    st.title("📊 Upcoming Matches")
    
    df = load_upcoming_matches()
    
    if df is not None and len(df) > 0:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            dates = sorted(df["match_date"].unique().to_list())
            selected_date = st.selectbox("Date", ["All"] + dates)
        
        with col2:
            tournaments = sorted(df["tournament_name"].drop_nulls().unique().to_list())
            selected_tournament = st.selectbox("Tournament", ["All"] + tournaments)
        
        with col3:
            odds_filter = st.selectbox("Odds Filter", ["All", "With Odds Only"])
        
        # Apply filters
        filtered = df
        if selected_date != "All":
            filtered = filtered.filter(pl.col("match_date") == selected_date)
        if selected_tournament != "All":
            filtered = filtered.filter(pl.col("tournament_name") == selected_tournament)
        if odds_filter == "With Odds Only":
            filtered = filtered.filter(pl.col("has_odds"))
        
        st.markdown(f"**{len(filtered)} matches**")
        
        # Display table
        display_cols = ["match_date", "match_time", "tournament_name", "player_name", 
                       "opponent_name", "odds_player", "odds_opponent", "round_name"]
        display_cols = [c for c in display_cols if c in filtered.columns]
        
        st.dataframe(
            filtered.select(display_cols).to_pandas(),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No upcoming matches loaded. Run:")
        st.code("python scripts/scrape_future.py --days 7")


# =============================================================================
# MODEL STATS PAGE
# =============================================================================

elif page == "📈 Model Stats":
    st.title("📈 Model Performance")
    
    model_info = load_model_info()
    hist_stats = load_historical_stats()
    
    # Model metrics
    if model_info:
        models = model_info.get("models", [])
        if models:
            latest = models[-1]
            metrics = latest.get("metrics", {})
            
            st.markdown("### Current Model Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Eval Accuracy", f"{metrics.get('eval_accuracy', 0)*100:.1f}%")
            with col2:
                st.metric("Eval AUC", f"{metrics.get('eval_auc', 0):.4f}")
            with col3:
                st.metric("Eval Log-Loss", f"{metrics.get('eval_logloss', 0):.4f}")
            with col4:
                st.metric("Eval Brier", f"{metrics.get('eval_brier', 0):.4f}")
            
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Train Accuracy", f"{metrics.get('train_accuracy', 0)*100:.1f}%")
            with col2:
                st.metric("Train AUC", f"{metrics.get('train_auc', 0):.4f}")
            with col3:
                st.metric("Version", latest.get("version", "N/A"))
            with col4:
                st.metric("Created", latest.get("created_at", "N/A")[:10])
    
    # Historical stats
    if hist_stats:
        st.markdown("### Training Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Matches", f"{hist_stats['total_matches']:,}")
        with col2:
            st.metric("Unique Players", f"{hist_stats['players']:,}")
        with col3:
            st.caption(f"Date Range: {hist_stats['date_range']}")
    
    # Odds range analysis
    st.markdown("### Performance by Odds Range")
    st.markdown("""
    Based on backtest analysis:
    
    | Odds Range | Edge | Recommendation |
    |------------|------|----------------|
    | 1.01-1.60 | Negative | ❌ Avoid |
    | 1.80-2.00 | +5.3% | ✅ Bet |
    | 2.50-3.00 | +6.4% | ✅ Bet |
    | 3.00-4.00 | +4.7% | ✅ Bet |
    | 4.00-5.00 | +7.9% | ✅ Bet |
    | 5.00+ | Negative | ❌ Avoid |
    """)


# =============================================================================
# PLAYER HISTORY PAGE
# =============================================================================

elif page == "👤 Player History":
    st.title("👤 Player History & Statistics")
    
    # Load data
    raw_matches = load_raw_matches()
    
    if raw_matches is None or len(raw_matches) == 0:
        st.warning("⚠️ No player data available. Run `python scripts/update_active_players.py` first.")
    else:
        st.success(f"✅ Loaded {len(raw_matches):,} historical matches")
        
        # Get unique players
        players_list = sorted(raw_matches["player_name"].unique().to_list())
        
        # Player selector
        selected_player = st.selectbox(
            "🔍 Select Player",
            options=players_list,
            index=0 if players_list else None
        )
        
        if selected_player:
            # Filter matches for selected player
            player_matches = raw_matches.filter(
                pl.col("player_name") == selected_player
            )
            
            # Calculate stats
            total_matches = len(player_matches)
            wins = player_matches.filter(pl.col("player_won") == True).height
            losses = total_matches - wins
            win_rate = wins / total_matches if total_matches > 0 else 0
            
            # Top row metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Matches", f"{total_matches}")
            
            with col2:
                st.metric("Wins", f"{wins}", delta=f"{win_rate:.1%}")
            
            with col3:
                st.metric("Losses", f"{losses}")
            
            with col4:
                st.metric("Win Rate", f"{win_rate:.1%}")
            
            st.markdown("---")
            
            # Surface performance
            st.subheader("🎾 Surface Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Surface stats table
                if "ground_type" in player_matches.columns:
                    surface_stats = player_matches.group_by("ground_type").agg([
                        pl.len().alias("matches"),
                        pl.col("player_won").sum().alias("wins")
                    ]).with_columns([
                        (pl.col("wins") / pl.col("matches") * 100).round(1).alias("win_rate")
                    ]).sort("matches", descending=True)
                    
                    st.dataframe(surface_stats, use_container_width=True, hide_index=True)
            
            with col2:
                # Surface win rate chart
                if "ground_type" in player_matches.columns and len(surface_stats) > 0:
                    import plotly.express as px
                    fig = px.bar(
                        surface_stats.to_pandas(),
                        x="ground_type",
                        y="win_rate",
                        title="Win Rate by Surface",
                        labels={"ground_type": "Surface", "win_rate": "Win Rate (%)"},
                        color="win_rate",
                        color_continuous_scale="RdYlGn"
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color="#ffffff"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Recent form
            st.subheader("📈 Recent Form (Last 20 Matches)")
            
            recent_matches = player_matches.sort("start_timestamp", descending=True).head(20)
            
            # Form string (W/L)
            form = "".join(["🟢" if row["player_won"] else "🔴" for row in recent_matches.iter_rows(named=True)])
            st.markdown(f"**Form:** {form}")
            
            # Recent matches win rate
            recent_wins = recent_matches.filter(pl.col("player_won") == True).height
            recent_rate = recent_wins / len(recent_matches) if len(recent_matches) > 0 else 0
            st.metric("Recent Win Rate (Last 20)", f"{recent_rate:.1%}")
            
            # Timeline chart
            if len(recent_matches) > 0:
                import plotly.graph_objects as go
                
                recent_df = recent_matches.to_pandas()
                recent_df['match_num'] = range(len(recent_df), 0, -1)
                recent_df['result'] = recent_df['player_won'].map({True: 'Win', False: 'Loss'})
                recent_df['color'] = recent_df['player_won'].map({True: '#00d97e', False: '#ff6b6b'})
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recent_df['match_num'],
                    y=recent_df['player_won'].astype(int),
                    mode='markers+lines',
                    marker=dict(
                        size=12,
                        color=recent_df['color'],
                        symbol='circle'
                    ),
                    line=dict(color='#4da6ff', width=2),
                    name='Match Result'
                ))
                
                fig.update_layout(
                    title="Recent 20 Matches Timeline",
                    xaxis_title="Matches Ago",
                    yaxis_title="Result",
                    yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Loss', 'Win']),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color="#ffffff",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # ELO Rating Progression
            st.subheader("📊 ELO Rating Progression")
            
            # Calculate ELO ratings over time
            def calculate_elo_progression(matches_df, k_factor=32, initial_elo=1500):
                """Calculate ELO rating progression over time."""
                # Sort by timestamp
                sorted_matches = matches_df.sort("start_timestamp").to_pandas()
                
                elo_ratings = []
                current_elo = initial_elo
                
                for idx, match in sorted_matches.iterrows():
                    # Assume opponent ELO is also 1500 average (simplified)
                    opponent_elo = 1500
                    
                    # Calculate expected score
                    expected = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
                    
                    # Actual score (1 for win, 0 for loss)
                    actual = 1 if match['player_won'] else 0
                    
                    # Update ELO
                    current_elo = current_elo + k_factor * (actual - expected)
                    
                    elo_ratings.append({
                        'match_num': idx + 1,
                        'elo': round(current_elo, 1),
                        'date': match.get('match_date', ''),
                        'opponent': match.get('opponent_name', 'Unknown'),
                        'result': 'Win' if match['player_won'] else 'Loss'
                    })
                
                return elo_ratings
            
            if len(player_matches) > 0:
                elo_data = calculate_elo_progression(player_matches)
                
                if elo_data:
                    import pandas as pd
                    elo_df = pd.DataFrame(elo_data)
                    
                    # Display current ELO
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current ELO", f"{elo_df['elo'].iloc[-1]:.0f}")
                    with col2:
                        st.metric("Peak ELO", f"{elo_df['elo'].max():.0f}")
                    with col3:
                        st.metric("Starting ELO", f"{elo_df['elo'].iloc[0]:.0f}")
                    
                    # ELO progression chart
                    import plotly.graph_objects as go
                    
                    fig2 = go.Figure()
                    
                    # Color points by win/loss
                    colors = ['#00d97e' if r == 'Win' else '#ff6b6b' for r in elo_df['result']]
                    
                    fig2.add_trace(go.Scatter(
                        x=elo_df['match_num'],
                        y=elo_df['elo'],
                        mode='lines+markers',
                        name='ELO Rating',
                        line=dict(color='#4da6ff', width=2),
                        marker=dict(
                            size=6,
                            color=colors,
                            line=dict(width=1, color='#ffffff')
                        ),
                        hovertemplate='<b>Match %{x}</b><br>' +
                                     'ELO: %{y:.0f}<br>' +
                                     '<extra></extra>'
                    ))
                    
                    # Add horizontal line at 1500 (average)
                    fig2.add_hline(
                        y=1500, 
                        line_dash="dash", 
                        line_color="#888888",
                        annotation_text="Average (1500)",
                        annotation_position="right"
                    )
                    
                    fig2.update_layout(
                        title="ELO Rating Over Time",
                        xaxis_title="Match Number",
                        yaxis_title="ELO Rating",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color="#ffffff",
                        height=450,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # ELO explanation
                    with st.expander("ℹ️ About ELO Ratings"):
                        st.markdown("""
                        **ELO Rating System:**
                        - Starting rating: 1500 (average)
                        - Green markers = Wins (ELO increases)
                        - Red markers = Losses (ELO decreases)
                        - Higher ELO = Stronger performance
                        
                        **Note:** This is a simplified calculation assuming all opponents have an average ELO of 1500.
                        Actual player ELO ratings would require tracking all players' ratings simultaneously.
                        """)
            
            st.markdown("---")
            
            # Head-to-head records
            st.subheader("🤝 Head-to-Head Records (Top 10 Opponents)")
            
            # Get all opponents
            opponents = player_matches["opponent_name"].value_counts().sort("count", descending=True).head(10)
            
            if len(opponents) > 0:
                h2h_data = []
                
                for row in opponents.iter_rows(named=True):
                    opp_name = row["opponent_name"]
                    opp_matches = player_matches.filter(pl.col("opponent_name") == opp_name)
                    opp_wins = opp_matches.filter(pl.col("player_won") == True).height
                    opp_total = len(opp_matches)
                    opp_losses = opp_total - opp_wins
                    
                    h2h_data.append({
                        "Opponent": opp_name,
                        "Matches": opp_total,
                        "Wins": opp_wins,
                        "Losses": opp_losses,
                        "Win Rate": f"{opp_wins/opp_total*100:.1f}%" if opp_total > 0 else "0%"
                    })
                
                import pandas as pd
                h2h_df = pd.DataFrame(h2h_data)
                st.dataframe(h2h_df, use_container_width=True, hide_index=True)
            else:
                st.info("No head-to-head data available")
            
            st.markdown("---")
            
            # Match history table
            st.subheader("📋 Recent Match History")
            
            # Select columns to display
            display_cols = ["match_date", "opponent_name", "player_won", "player_sets", "opponent_sets", 
                          "tournament_name", "ground_type"]
            
            available_cols = [col for col in display_cols if col in recent_matches.columns]
            
            if available_cols:
                history_df = recent_matches.select(available_cols).to_pandas()
                history_df['player_won'] = history_df['player_won'].map({True: '✅ Win', False: '❌ Loss'})
                
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "match_date": "Date",
                        "opponent_name": "Opponent",
                        "player_won": "Result",
                        "player_sets": "Sets Won",
                        "opponent_sets": "Sets Lost",
                        "tournament_name": "Tournament",
                        "ground_type": "Surface"
                    }
                )


# =============================================================================
# MODEL STATS PAGE
# =============================================================================


# =============================================================================
# BETTING LOG PAGE
# =============================================================================

elif page == "💰 Betting Log":
    st.title("💰 Betting Log")
    
    bet_log = load_bet_log()
    bets = bet_log.get("bets", [])
    
    # Summary stats
    stats = get_bet_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Bets", stats["total"])
        with col2:
            st.metric("Pending", stats["pending"])
        with col3:
            win_rate_str = f"{stats['win_rate']:.1%}" if stats["won"] + stats["lost"] > 0 else "N/A"
            st.metric("Win Rate", win_rate_str, delta=f"{stats['won']}W / {stats['lost']}L")
        with col4:
            st.metric("Total Profit", f"${stats['total_profit']:.2f}", 
                     delta=f"ROI: {stats['roi']:.1%}" if stats["total_staked"] > 0 else "-")
        
        st.markdown("---")
    
    # Auto-check outcomes button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("🔄 Auto-Check Outcomes", use_container_width=True, help="Fetch match results from SofaScore"):
            with st.spinner("Checking pending bets..."):
                updated = auto_check_outcomes()
                if updated > 0:
                    st.success(f"✅ Updated {updated} bet(s)!")
                    st.rerun()
                else:
                    st.info("No pending bets to update or matches not finished yet.")
    
    if bets:
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            filter_outcome = st.selectbox("Filter by Outcome", ["All", "Pending", "Won", "Lost"])
        with col2:
            sort_order = st.selectbox("Sort by", ["Most Recent", "Oldest First", "Highest Stake", "Highest Edge"])
        
        # Apply filters
        filtered_bets = bets.copy()
        if filter_outcome != "All":
            filtered_bets = [b for b in filtered_bets if b.get("outcome") == filter_outcome]
        
        # Apply sorting
        if sort_order == "Most Recent":
            filtered_bets.sort(key=lambda x: x.get("placed_at", ""), reverse=True)
        elif sort_order == "Oldest First":
            filtered_bets.sort(key=lambda x: x.get("placed_at", ""))
        elif sort_order == "Highest Stake":
            filtered_bets.sort(key=lambda x: x.get("stake", 0), reverse=True)
        elif sort_order == "Highest Edge":
            filtered_bets.sort(key=lambda x: x.get("edge", 0), reverse=True)
        
        st.markdown(f"**Showing {len(filtered_bets)} bets**")
        st.markdown("---")
        
        # Display bets
        for bet in filtered_bets:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    outcome_icon = {"Pending": "⏳", "Won": "✅", "Lost": "❌"}.get(bet["outcome"], "❓")
                    st.markdown(f"### {outcome_icon} **{bet.get('player', 'Unknown')}** vs {bet.get('opponent', 'Unknown')}")
                    st.caption(f"🏆 {bet.get('tournament', 'Unknown')} | 📅 {bet.get('match_date', '')} {bet.get('match_time', '')}")
                
                with col2:
                    st.metric("Odds", f"{bet.get('odds', 0):.2f}")
                    st.metric("Stake", f"${bet.get('stake', 0):.2f}")
                
                with col3:
                    st.metric("Edge", f"+{bet.get('edge', 0):.1f}%")
                    st.metric("Model", f"{bet.get('model_prob', 0):.1f}%")
                
                with col4:
                    # Outcome management
                    if bet["outcome"] == "Pending":
                        st.markdown("**Update:**")
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button("✅", key=f"won_{bet['id']}", help="Mark as Won"):
                                update_bet_outcome(bet["id"], "Won")
                                st.rerun()
                        with btn_col2:
                            if st.button("❌", key=f"lost_{bet['id']}", help="Mark as Lost"):
                                update_bet_outcome(bet["id"], "Lost")
                                st.rerun()
                    else:
                        profit = bet.get("profit", 0)
                        profit_str = f"+${profit:.2f}" if profit >= 0 else f"-${abs(profit):.2f}"
                        st.metric("Profit", profit_str)
                
                st.markdown("---")
        
        # Export option
        st.markdown("### 📥 Export")
        st.download_button(
            "Download Bet Log",
            data=json.dumps(bet_log, indent=2, default=str),
            file_name="bet_log_export.json",
            mime="application/json"
        )
    else:
        st.info("No bets logged yet. Save bets from the Value Bets page!")
        st.markdown("""
        **How to log bets:**
        1. Go to **🎯 Value Bets** page
        2. Find a bet you want to track
        3. Set your stake amount
        4. Click **💾 Save Bet**
        """)


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("Tennis Betting AI | Built with ❤️ and XGBoost")
