"""Rule-based agent using simple heuristics."""

import random
from typing import List, Optional

from app.game.state import GameState


class RuleBasedAgent:
    """Agent using simple heuristics.

    Strategy:
    1. Take winning move if available
    2. Block opponent's winning move
    3. Prefer center, then corners, then edges
    """

    def __init__(self):
        """Initialize rule-based agent."""
        self.name = "Rule-Based"

    def select_action(self, state: GameState, player_id: int) -> int:
        """Select move using heuristic rules.

        Args:
            state: Current game state.
            player_id: Player ID (1 for X, -1 for O).

        Returns:
            Move index (0-8).
        """
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")

        # 1. Check for winning move
        for move in valid_moves:
            test_state = state.apply_move(move, player_id)
            if self._check_winner(test_state, player_id):
                return move

        # 2. Block opponent's winning move
        opponent = -player_id
        for move in valid_moves:
            test_state = state.apply_move(move, opponent)
            if self._check_winner(test_state, opponent):
                return move

        # 3. Prefer center, then corners, then edges
        center = 4
        corners = [0, 2, 6, 8]
        edges = [1, 3, 5, 7]

        if center in valid_moves:
            return center

        available_corners = [m for m in corners if m in valid_moves]
        if available_corners:
            return random.choice(available_corners)

        available_edges = [m for m in edges if m in valid_moves]
        if available_edges:
            return random.choice(available_edges)

        return random.choice(valid_moves)

    def _check_winner(self, state: GameState, player_id: int) -> bool:
        """Check if player has won in given state.

        Args:
            state: Game state to check.
            player_id: Player ID to check.

        Returns:
            True if player has won, False otherwise.
        """
        board = state.board

        # Check rows
        for row in range(3):
            if all(board[row, col] == player_id for col in range(3)):
                return True

        # Check columns
        for col in range(3):
            if all(board[row, col] == player_id for row in range(3)):
                return True

        # Check diagonals
        if all(board[i, i] == player_id for i in range(3)):
            return True
        if all(board[i, 2 - i] == player_id for i in range(3)):
            return True

        return False

