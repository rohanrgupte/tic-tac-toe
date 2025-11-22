"""Random agent that selects moves uniformly at random."""

import random
from typing import List

from app.game.state import GameState


class RandomAgent:
    """Agent that picks any legal move uniformly at random.

    Used as a simple baseline for comparison.
    """

    def __init__(self):
        """Initialize random agent."""
        self.name = "Random"

    def select_action(self, state: GameState, player_id: int) -> int:
        """Select a random valid move.

        Args:
            state: Current game state.
            player_id: Player ID (1 for X, -1 for O). Not used by this agent.

        Returns:
            Move index (0-8).
        """
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        return random.choice(valid_moves)

