"""
Tournament Showdown Mode - Bracket Simulation and Comparison.

Simulates tournament brackets, runs model predictions through each round,
and compares against actual results with visualization.
"""
from .bracket import TournamentBracket, TournamentConfig, BracketMatch
from .simulator import TournamentSimulator, ShowdownStats
from .visualizer import BracketVisualizer

__all__ = [
    "TournamentBracket",
    "TournamentConfig", 
    "BracketMatch",
    "TournamentSimulator",
    "ShowdownStats",
    "BracketVisualizer",
]
