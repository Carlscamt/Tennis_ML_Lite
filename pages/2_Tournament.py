"""
🏆 Tournament Bracket Page - Fixed match linkage
"""
import streamlit as st
import streamlit.components.v1 as components
import sys
from pathlib import Path
from collections import defaultdict

st.set_page_config(page_title="Tournament Bracket", page_icon="🏆", layout="wide")

st.title("🏆 Tournament Bracket")
st.markdown("---")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

def get_tournaments():
    try:
        import importlib.util
        _path = ROOT / "src" / "scraper.py"
        _spec = importlib.util.spec_from_file_location("scraper_mod", _path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        return _mod.fetch_ongoing_tournaments(), _mod
    except Exception as e:
        st.error(f"Error: {e}")
        return [], None

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("📋 Load Tournaments", type="primary"):
        with st.spinner("Fetching..."):
            tournaments, scraper = get_tournaments()
            st.session_state["tournaments"] = tournaments
            st.session_state["scraper"] = scraper

if "tournaments" in st.session_state and st.session_state["tournaments"]:
    tournaments = st.session_state["tournaments"]
    options = {f"{t['name']} ({t['category']})": t for t in tournaments}
    selected = st.selectbox("Select Tournament", list(options.keys()))
    tournament = options[selected]
    
    if st.button("📊 Load Bracket"):
        with st.spinner("Fetching matches..."):
            try:
                scraper = st.session_state.get("scraper")
                if scraper:
                    matches = scraper.fetch_tournament_matches(tournament['id'], tournament.get('season_id'))
                    st.session_state["matches"] = matches
                    st.session_state["tournament_name"] = tournament['name']
            except Exception as e:
                st.error(f"Error: {e}")

if "matches" in st.session_state and st.session_state["matches"]:
    matches = st.session_state["matches"]
    st.subheader(f"📍 {st.session_state.get('tournament_name', 'Tournament')}")
    
    # Group matches by round
    rounds = defaultdict(list)
    for m in matches:
        round_info = m.get("roundInfo", {})
        round_name = round_info.get("name", "Unknown")
        rounds[round_name].append(m)
    
    round_order = ["128", "64", "32", "16", "Quarter", "Semi", "Final"]
    
    def round_sort_key(r):
        r_lower = r.lower()
        for i, pattern in enumerate(round_order):
            if pattern.lower() in r_lower:
                return i
        return 99
    
    sorted_rounds = sorted(rounds.keys(), key=round_sort_key)
    
    # Build bracket tree by finding which matches feed into which
    # For each match in round N, find the match in round N+1 that contains one of its players
    
    def get_player_ids(match):
        """Get both player IDs from a match"""
        home_id = match.get("homeTeam", {}).get("id")
        away_id = match.get("awayTeam", {}).get("id")
        return (home_id, away_id)
    
    def get_winner_id(match):
        """Get winner player ID"""
        winner_code = match.get("winnerCode")
        if winner_code == 1:
            return match.get("homeTeam", {}).get("id")
        elif winner_code == 2:
            return match.get("awayTeam", {}).get("id")
        return None
    
    # Create player -> match mapping for each round
    round_player_match = {}  # {round_name: {player_id: match}}
    for round_name, round_matches in rounds.items():
        round_player_match[round_name] = {}
        for m in round_matches:
            home_id, away_id = get_player_ids(m)
            if home_id:
                round_player_match[round_name][home_id] = m
            if away_id:
                round_player_match[round_name][away_id] = m
    
    # Build bracket structure: for each match, find its "children" (previous round matches)
    match_children = {}  # match_id -> [child_match_1, child_match_2]
    
    for i, round_name in enumerate(sorted_rounds[1:], 1):  # Skip first round
        prev_round = sorted_rounds[i-1]
        for match in rounds[round_name]:
            home_id, away_id = get_player_ids(match)
            children = []
            
            # Find which previous match had home player
            if home_id and home_id in round_player_match.get(prev_round, {}):
                children.append(round_player_match[prev_round][home_id])
            
            # Find which previous match had away player  
            if away_id and away_id in round_player_match.get(prev_round, {}):
                children.append(round_player_match[prev_round][away_id])
            
            match_children[match.get("id")] = children
    
    # Now order matches in each round based on bracket position
    # Start from final and work backwards to assign positions
    
    def assign_bracket_positions(sorted_rounds, rounds, match_children):
        """Assign position indices to matches for proper bracket ordering"""
        positions = {}  # match_id -> position
        
        # Final gets position 0
        if sorted_rounds:
            final_round = sorted_rounds[-1]
            for idx, m in enumerate(rounds[final_round]):
                positions[m.get("id")] = idx
        
        # Work backwards
        for i in range(len(sorted_rounds) - 2, -1, -1):
            round_name = sorted_rounds[i]
            next_round = sorted_rounds[i + 1]
            
            # For each match in next round, its children get positions 2*pos and 2*pos+1
            match_to_pos = []
            for m in rounds[round_name]:
                m_id = m.get("id")
                # Find which next-round match this feeds into
                parent_pos = 999
                for next_m in rounds[next_round]:
                    if m_id in [c.get("id") for c in match_children.get(next_m.get("id"), [])]:
                        parent_pos = positions.get(next_m.get("id"), 999)
                        children = match_children.get(next_m.get("id"), [])
                        is_first_child = children and children[0].get("id") == m_id
                        child_offset = 0 if is_first_child else 1
                        positions[m_id] = parent_pos * 2 + child_offset
                        break
                if m_id not in positions:
                    positions[m_id] = len(match_to_pos)
                match_to_pos.append((positions.get(m_id, 999), m))
            
        return positions
    
    positions = assign_bracket_positions(sorted_rounds, rounds, match_children)
    
    # Order each round by position
    ordered_rounds = {}
    for round_name in sorted_rounds:
        ordered_rounds[round_name] = sorted(
            rounds[round_name], 
            key=lambda m: positions.get(m.get("id"), 999)
        )
    
    # SVG Bracket Generator
    match_height = 50
    match_width = 180
    gap_horizontal = 60
    connector_width = 30
    
    num_rounds = len(sorted_rounds)
    max_matches_first = len(ordered_rounds.get(sorted_rounds[0], [])) if sorted_rounds else 8
    
    svg_width = num_rounds * (match_width + gap_horizontal + connector_width) + 100
    svg_height = max(max_matches_first * (match_height + 20) + 100, 400)
    
    svg_content = f'''
    <svg width="{svg_width}" height="{svg_height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="matchBg" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#1a1a2e"/>
                <stop offset="100%" style="stop-color:#16213e"/>
            </linearGradient>
        </defs>
        <style>
            .match-box {{ fill: url(#matchBg); stroke: #0f3460; stroke-width: 1; }}
            .match-box-winner {{ fill: url(#matchBg); stroke: #00d26a; stroke-width: 2; }}
            .player-text {{ font-family: Arial, sans-serif; font-size: 11px; fill: #fafafa; }}
            .player-winner {{ font-family: Arial, sans-serif; font-size: 11px; fill: #00d26a; font-weight: bold; }}
            .player-loser {{ font-family: Arial, sans-serif; font-size: 11px; fill: #666; }}
            .score-text {{ font-family: Arial, sans-serif; font-size: 10px; fill: #888; }}
            .round-title {{ font-family: Arial, sans-serif; font-size: 12px; fill: #667eea; font-weight: bold; }}
            .connector {{ stroke: #3a3a5e; stroke-width: 2; fill: none; }}
        </style>
    '''
    
    for round_idx, round_name in enumerate(sorted_rounds):
        round_matches = ordered_rounds[round_name][:16]
        num_matches = len(round_matches)
        
        # Calculate vertical spacing
        if round_idx == 0:
            vertical_gap = match_height + 20
            start_y = 50
        else:
            factor = 2 ** round_idx
            vertical_gap = (match_height + 20) * factor
            start_y = 50 + (vertical_gap - match_height) / 2
        
        x = 50 + round_idx * (match_width + gap_horizontal + connector_width)
        
        svg_content += f'<text x="{x + match_width/2}" y="30" text-anchor="middle" class="round-title">{round_name}</text>'
        
        for match_idx, match in enumerate(round_matches):
            y = start_y + match_idx * vertical_gap
            
            home = match.get("homeTeam", {})
            away = match.get("awayTeam", {})
            home_name = home.get("name", "TBD")[:18]
            away_name = away.get("name", "TBD")[:18]
            
            home_score = str(match.get("homeScore", {}).get("current", ""))
            away_score = str(match.get("awayScore", {}).get("current", ""))
            
            winner_code = match.get("winnerCode")
            
            home_class = "player-winner" if winner_code == 1 else ("player-loser" if winner_code == 2 else "player-text")
            away_class = "player-winner" if winner_code == 2 else ("player-loser" if winner_code == 1 else "player-text")
            box_class = "match-box-winner" if winner_code else "match-box"
            
            svg_content += f'<rect x="{x}" y="{y}" width="{match_width}" height="{match_height}" rx="4" class="{box_class}"/>'
            svg_content += f'<text x="{x + 8}" y="{y + 18}" class="{home_class}">{home_name}</text>'
            svg_content += f'<text x="{x + 8}" y="{y + 38}" class="{away_class}">{away_name}</text>'
            svg_content += f'<text x="{x + match_width - 25}" y="{y + 18}" class="score-text">{home_score}</text>'
            svg_content += f'<text x="{x + match_width - 25}" y="{y + 38}" class="score-text">{away_score}</text>'
            svg_content += f'<line x1="{x}" y1="{y + match_height/2}" x2="{x + match_width}" y2="{y + match_height/2}" stroke="#0f3460"/>'
            
            # Draw connectors
            if round_idx < num_rounds - 1 and match_idx % 2 == 0:
                top_y = y + match_height / 2
                if match_idx + 1 < num_matches:
                    bottom_match_y = start_y + (match_idx + 1) * vertical_gap
                    bottom_y = bottom_match_y + match_height / 2
                else:
                    bottom_y = top_y
                
                mid_y = (top_y + bottom_y) / 2
                conn_x1 = x + match_width
                conn_x2 = conn_x1 + 15
                next_x = x + match_width + connector_width + gap_horizontal
                
                svg_content += f'<path d="M{conn_x1},{top_y} L{conn_x2},{top_y} L{conn_x2},{mid_y} L{next_x},{mid_y}" class="connector"/>'
                svg_content += f'<path d="M{conn_x1},{bottom_y} L{conn_x2},{bottom_y} L{conn_x2},{mid_y}" class="connector"/>'
    
    svg_content += '</svg>'
    
    html_content = f'''<!DOCTYPE html><html><head><style>body {{ background: #0e1117; margin: 0; padding: 10px; overflow-x: auto; }}</style></head><body>{svg_content}</body></html>'''
    
    components.html(html_content, height=int(svg_height) + 50, scrolling=True)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    finished = sum(1 for m in matches if m.get("status", {}).get("type") == "finished")
    with col1:
        st.metric("Total", len(matches))
    with col2:
        st.metric("Finished", finished)
    with col3:
        st.metric("Upcoming", len(matches) - finished)
else:
    st.info("👆 Click 'Load Tournaments' then select a tournament.")
