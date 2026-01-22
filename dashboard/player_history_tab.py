

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
            
            # Head-to-head records
            st.subheader("🤝 Head-to-Head Records")
            
            # Get all opponents
            opponents = player_matches["opponent_name"].value_counts().sort("counts", descending=True).head(10)
            
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
                        "tournament_name": " Tournament",
                        "ground_type": "Surface"
                    }
                )

