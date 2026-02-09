"""
Tennis Prediction Dashboard - Streamlit App
============================================
Run with: streamlit run app.py
"""
import streamlit as st

# Page config
st.set_page_config(
    page_title="Tennis Predictions",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #0f3460;
    }
    .stDataFrame {
        border-radius: 10px;
    }
    .positive-edge {
        color: #00d26a;
        font-weight: bold;
    }
    .negative-edge {
        color: #ff6b6b;
    }
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Main page content
st.title("🎾 Tennis Prediction System")
st.markdown("---")

# Navigation info
st.markdown("""
### Welcome to the Tennis Prediction Dashboard

Use the sidebar to navigate between pages:

- **🎯 Predictions** - View upcoming value bets with edge calculations
- **🏆 Tournament** - Analyze ongoing tournaments match by match  
- **📊 Performance** - Model metrics and backtest results

---
""")

# Quick stats from cached data
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Active Model", "v1.0.3", "Challenger")
    
with col2:
    st.metric("Data Coverage", "2,500+", "matches")
    
with col3:
    st.metric("Last Updated", "Today", "scraped")

st.markdown("---")

# Quick start
st.markdown("""
### Quick Start

```bash
# List ongoing tournaments
python tennis.py tournaments

# Analyze a tournament
python tennis.py analyze --tournament <ID>

# Get predictions
python tennis.py predict --days 3
```
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit • Data from SofaScore • Models trained with XGBoost")
