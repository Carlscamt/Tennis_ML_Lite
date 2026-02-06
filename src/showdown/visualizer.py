"""
Tournament Bracket Visualizer.

Generates beautiful, interactive tournament bracket visualizations
comparing model predictions against actual results.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import logging

from .bracket import TournamentBracket, BracketMatch
from .simulator import ShowdownStats

logger = logging.getLogger(__name__)


# Try to import plotly, fall back to basic HTML if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed. Using basic HTML visualization.")


class BracketVisualizer:
    """
    Generate tournament bracket visualizations.
    
    Supports:
    - Interactive HTML with Plotly (if available)
    - Basic HTML fallback
    - JSON export for custom rendering
    """
    
    COLORS = {
        "correct": "#22c55e",      # Green
        "incorrect": "#ef4444",     # Red
        "pending": "#6b7280",       # Gray
        "background": "#1f2937",    # Dark
        "card": "#374151",          # Card background
        "text": "#f9fafb",          # Light text
        "accent": "#3b82f6",        # Blue accent
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for output files (defaults to results/)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def render_html(
        self, 
        bracket: TournamentBracket, 
        stats: ShowdownStats,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Render interactive HTML bracket visualization.
        
        Args:
            bracket: Tournament bracket with predictions
            stats: Showdown statistics
            output_path: Output file path
            
        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            output_path = self.output_dir / f"showdown_{bracket.config.name.replace(' ', '_')}_{bracket.config.year}.html"
        
        output_path = Path(output_path)
        
        if PLOTLY_AVAILABLE:
            html_content = self._render_plotly_bracket(bracket, stats)
        else:
            html_content = self._render_basic_html(bracket, stats)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Saved visualization to {output_path}")
        return output_path
    
    def _render_plotly_bracket(
        self, 
        bracket: TournamentBracket, 
        stats: ShowdownStats
    ) -> str:
        """Generate Plotly-based interactive visualization."""
        
        # Create figure with subplots: bracket + stats
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.75, 0.25],
            specs=[[{"type": "scatter"}, {"type": "bar"}]],
            subplot_titles=[
                f"🎾 {bracket.config} - Tournament Bracket",
                "Accuracy by Round"
            ]
        )
        
        # Bracket visualization
        rounds = sorted(bracket.rounds.keys())
        max_matches_per_round = max(len(bracket.rounds[r]) for r in rounds) if rounds else 0
        
        x_positions = []
        y_positions = []
        colors = []
        texts = []
        hovertexts = []
        
        for round_num in rounds:
            matches = bracket.rounds[round_num]
            num_matches = len(matches)
            
            # Spread matches vertically
            y_spacing = max_matches_per_round / (num_matches + 1)
            
            for i, match in enumerate(matches):
                x = round_num * 2
                y = (i + 1) * y_spacing
                
                x_positions.append(x)
                y_positions.append(y)
                
                # Color based on prediction correctness
                if match.prediction_correct is True:
                    color = self.COLORS["correct"]
                elif match.prediction_correct is False:
                    color = self.COLORS["incorrect"]
                else:
                    color = self.COLORS["pending"]
                
                colors.append(color)
                
                # Display text
                winner = match.model_winner_name or "TBD"
                confidence = match.model_confidence * 100 if match.model_confidence else 0
                texts.append(f"{winner[:12]}...")
                
                # Hover text
                hover = (
                    f"<b>{match.round_name}</b><br>"
                    f"{match.player1_name} vs {match.player2_name}<br>"
                    f"<br>"
                    f"<b>Model Pick:</b> {match.model_winner_name} ({confidence:.0f}%)<br>"
                    f"<b>Actual:</b> {match.actual_winner_name or 'TBD'}<br>"
                    f"<b>Result:</b> {'✓ Correct' if match.prediction_correct else '✗ Wrong' if match.prediction_correct is False else 'Pending'}"
                )
                hovertexts.append(hover)
        
        # Add bracket scatter plot
        fig.add_trace(
            go.Scatter(
                x=x_positions,
                y=y_positions,
                mode="markers+text",
                marker=dict(
                    size=30,
                    color=colors,
                    line=dict(width=2, color="white"),
                    symbol="square"
                ),
                text=texts,
                textposition="middle center",
                textfont=dict(size=8, color="white"),
                hovertext=hovertexts,
                hoverinfo="text",
                name="Matches"
            ),
            row=1, col=1
        )
        
        # Add round labels
        for round_num in rounds:
            round_name = bracket.rounds[round_num][0].round_name if bracket.rounds[round_num] else f"R{round_num}"
            fig.add_annotation(
                x=round_num * 2,
                y=max_matches_per_round + 1,
                text=f"<b>{round_name}</b>",
                showarrow=False,
                font=dict(size=12, color="white"),
                row=1, col=1
            )
        
        # Accuracy bar chart
        round_names = list(stats.accuracy_by_round.keys())
        accuracies = list(stats.accuracy_by_round.values())
        
        bar_colors = [
            self.COLORS["correct"] if acc >= 0.6 else 
            self.COLORS["accent"] if acc >= 0.5 else 
            self.COLORS["incorrect"] 
            for acc in accuracies
        ]
        
        fig.add_trace(
            go.Bar(
                x=round_names,
                y=[acc * 100 for acc in accuracies],
                marker_color=bar_colors,
                text=[f"{acc*100:.0f}%" for acc in accuracies],
                textposition="outside",
                name="Accuracy"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            height=700,
            showlegend=False,
            title=dict(
                text=f"🏆 {bracket.config} Showdown | Overall Accuracy: {stats.accuracy*100:.1f}%",
                font=dict(size=20)
            ),
            paper_bgcolor=self.COLORS["background"],
            plot_bgcolor=self.COLORS["background"],
        )
        
        # Hide axes for bracket
        fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, showticklabels=False, row=1, col=1)
        
        # Style bar chart
        fig.update_yaxes(title="Accuracy (%)", range=[0, 100], row=1, col=2)
        fig.update_xaxes(tickangle=45, row=1, col=2)
        
        # Add summary annotation
        summary_text = (
            f"<b>Summary</b><br>"
            f"Total Matches: {stats.total_matches}<br>"
            f"Correct: {stats.correct_predictions}/{stats.predicted_matches}<br>"
            f"Upsets Caught: {stats.upsets_predicted}/{stats.upsets_total}"
        )
        
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text=summary_text,
            showarrow=False,
            font=dict(size=12),
            align="center",
            bgcolor=self.COLORS["card"],
            bordercolor=self.COLORS["accent"],
            borderwidth=1,
            borderpad=10,
        )
        
        return fig.to_html(full_html=True, include_plotlyjs=True)
    
    def _render_basic_html(
        self, 
        bracket: TournamentBracket, 
        stats: ShowdownStats
    ) -> str:
        """Generate simple bracket-style HTML visualization."""
        
        # Limit to last 4 rounds for cleaner visualization (QF, SF, F or similar)
        all_rounds = sorted(bracket.rounds.keys())
        display_rounds = all_rounds[-4:] if len(all_rounds) > 4 else all_rounds
        
        rounds_html = []
        for round_num in display_rounds:
            matches = bracket.rounds[round_num]
            if not matches:
                continue
            
            round_name = matches[0].round_name
            matches_html = []
            
            for match in matches:
                # Determine colors
                if match.prediction_correct is True:
                    border_color = "#22c55e"  # green
                    status = "CORRECT"
                elif match.prediction_correct is False:
                    border_color = "#ef4444"  # red
                    status = "WRONG"
                else:
                    border_color = "#6b7280"  # gray
                    status = "TBD"
                
                # Highlight winner
                p1_style = "font-weight:bold; color:#22c55e;" if match.actual_winner_id == match.player1_id else ""
                p2_style = "font-weight:bold; color:#22c55e;" if match.actual_winner_id == match.player2_id else ""
                
                # Model pick indicator
                p1_pick = " ◀" if match.model_winner_id == match.player1_id else ""
                p2_pick = " ◀" if match.model_winner_id == match.player2_id else ""
                
                confidence = f"{match.model_confidence*100:.0f}%" if match.model_confidence else "-"
                
                matches_html.append(f'''
                <div class="match" style="border-left: 4px solid {border_color};">
                    <div class="player" style="{p1_style}">{match.player1_name}{p1_pick}</div>
                    <div class="player" style="{p2_style}">{match.player2_name}{p2_pick}</div>
                    <div class="meta">{confidence} • {status}</div>
                </div>
                ''')
            
            rounds_html.append(f'''
            <div class="round">
                <div class="round-header">{round_name}</div>
                <div class="matches">
                    {''.join(matches_html)}
                </div>
            </div>
            ''')
        
        # Stats table
        stats_rows = []
        for round_name, accuracy in stats.accuracy_by_round.items():
            color = "#22c55e" if accuracy >= 0.7 else "#f59e0b" if accuracy >= 0.5 else "#ef4444"
            stats_rows.append(f'<tr><td>{round_name}</td><td style="color:{color}">{accuracy*100:.0f}%</td></tr>')
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{bracket.config} Bracket</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 20px;
            min-height: 100vh;
        }}
        h1 {{
            text-align: center;
            font-size: 1.8em;
            margin-bottom: 5px;
        }}
        .subtitle {{
            text-align: center;
            font-size: 1.2em;
            color: #3b82f6;
            margin-bottom: 30px;
        }}
        .bracket {{
            display: flex;
            justify-content: center;
            gap: 10px;
            overflow-x: auto;
            padding: 20px 0;
        }}
        .round {{
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            min-width: 180px;
        }}
        .round-header {{
            text-align: center;
            font-weight: bold;
            padding: 8px;
            background: #1e293b;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 0.85em;
        }}
        .matches {{
            display: flex;
            flex-direction: column;
            justify-content: space-around;
            flex: 1;
            gap: 8px;
        }}
        .match {{
            background: #1e293b;
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 0.85em;
        }}
        .player {{
            padding: 3px 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .meta {{
            font-size: 0.75em;
            color: #64748b;
            margin-top: 4px;
            text-align: right;
        }}
        .stats {{
            max-width: 400px;
            margin: 30px auto 0;
            background: #1e293b;
            border-radius: 10px;
            padding: 20px;
        }}
        .stats h2 {{
            font-size: 1em;
            margin-bottom: 15px;
            text-align: center;
        }}
        .stats table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .stats td {{
            padding: 6px 10px;
            border-bottom: 1px solid #334155;
        }}
        .stats td:last-child {{
            text-align: right;
            font-weight: bold;
        }}
        .summary {{
            display: flex;
            justify-content: space-around;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #334155;
            font-size: 0.9em;
        }}
        .summary div {{
            text-align: center;
        }}
        .summary strong {{
            display: block;
            font-size: 1.3em;
            color: #3b82f6;
        }}
    </style>
</head>
<body>
    <h1>{bracket.config}</h1>
    <div class="subtitle">{stats.accuracy*100:.1f}% Accuracy ({stats.correct_predictions}/{stats.predicted_matches})</div>
    
    <div class="bracket">
        {''.join(rounds_html)}
    </div>
    
    <div class="stats">
        <h2>Accuracy by Round</h2>
        <table>
            {''.join(stats_rows)}
        </table>
        <div class="summary">
            <div><strong>{stats.total_matches}</strong>Matches</div>
            <div><strong>{stats.correct_predictions}</strong>Correct</div>
            <div><strong>{stats.upsets_predicted}/{stats.upsets_total}</strong>Upsets</div>
            <div><strong>{stats.avg_confidence*100:.0f}%</strong>Avg Conf</div>
        </div>
    </div>
</body>
</html>
'''
        return html
    
    def export_json(
        self, 
        bracket: TournamentBracket, 
        stats: ShowdownStats,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export bracket and stats as JSON for custom rendering.
        
        Args:
            bracket: Tournament bracket
            stats: Showdown statistics
            output_path: Output file path
            
        Returns:
            Path to JSON file
        """
        if output_path is None:
            output_path = self.output_dir / f"showdown_{bracket.config.name.replace(' ', '_')}_{bracket.config.year}.json"
        
        output_path = Path(output_path)
        
        data = {
            "bracket": bracket.to_dict(),
            "stats": stats.to_dict(),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported JSON to {output_path}")
        return output_path
