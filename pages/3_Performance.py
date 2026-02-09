"""
📊 Model Performance Page
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

st.set_page_config(page_title="Performance", page_icon="📊", layout="wide")

st.title("📊 Model Performance")
st.markdown("---")

# Paths
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

# Load model registry
@st.cache_data
def load_registry():
    registry_path = MODELS_DIR / "registry.json"
    if registry_path.exists():
        with open(registry_path) as f:
            return json.load(f)
    return {"versions": []}

registry = load_registry()

# Show model versions
st.subheader("📦 Model Registry")

if registry.get("versions"):
    versions = registry["versions"]
    
    for v in versions[-5:]:  # Show last 5
        version = v.get("version", "?")
        stage = v.get("stage", "Unknown")
        created = v.get("created_at", "")[:10]
        
        stage_color = {"Production": "🟢", "Challenger": "🟡", "Staging": "🔵", "Archived": "⚪"}.get(stage, "⚫")
        
        st.markdown(f"{stage_color} **{version}** - {stage} _(created {created})_")
else:
    st.info("No models registered yet.")

st.markdown("---")

# Load backtest results
@st.cache_data
def load_backtest_results():
    results = []
    if RESULTS_DIR.exists():
        for f in RESULTS_DIR.glob("*.json"):
            try:
                with open(f) as file:
                    data = json.load(f)
                    data["file"] = f.name
                    results.append(data)
            except:
                pass
    return results

results = load_backtest_results()

st.subheader("📈 Backtest Results")

if results:
    # Find summary metrics if available
    for r in results:
        if "roi" in r or "total_roi" in r:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                roi = r.get("roi", r.get("total_roi", 0))
                st.metric("ROI", f"{roi*100:.1f}%" if isinstance(roi, float) else str(roi))
            with col2:
                st.metric("Win Rate", f"{r.get('win_rate', 0)*100:.1f}%")
            with col3:
                st.metric("Total Bets", r.get("total_bets", r.get("num_bets", "?")))
            with col4:
                st.metric("Profit", f"${r.get('profit', r.get('total_profit', 0)):.0f}")
            break
    
    # Equity curve if available
    for r in results:
        if "equity_curve" in r:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=r["equity_curve"],
                mode="lines",
                name="Bankroll",
                line=dict(color="#667eea", width=2)
            ))
            fig.update_layout(
                title="Equity Curve",
                yaxis_title="Bankroll ($)",
                xaxis_title="Bet #",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
            break
else:
    st.info("No backtest results found. Run: `python tennis.py backtest`")

st.markdown("---")

# Quick actions
st.subheader("🚀 Quick Actions")
st.code("""
# Run backtest
python tennis.py backtest

# Train new model
python tennis.py train

# Promote model to production
python tennis.py promote v1.0.3 Production
""")
