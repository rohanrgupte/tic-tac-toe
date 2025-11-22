"""Game environment with rules, state transitions, and terminal detection."""

from typing import Optional, Tuple

import numpy as np

from app.game.state import GameState


class GameEnvironment:
    """Tic-Tac-Toe game environment.

    Manages game rules, state transitions, and win/draw detection.
    """

    def __init__(self):
        """Initialize empty game environment."""
        self.reset()

    def reset(self) -> GameState:
        """Reset to initial empty board state.

        Returns:
            Initial GameState.
        """
        self.state = GameState()
        self.current_player = 1  # X starts
        return self.state

    def step(self, move: int) -> Tuple[GameState, Optional[int], bool]:
        """Apply a move and update game state.

        Args:
            move: Move index (0-8).

        Returns:
            Tuple of (new_state, reward, done):
            - new_state: GameState after move
            - reward: 1 if current player wins, -1 if loses, 0 if draw/ongoing
            - done: True if game is over, False otherwise

        Raises:
            ValueError: If move is invalid.
        """
        if not self.state.is_valid_move(move):
            raise ValueError(f"Invalid move: {move}")

        self.state = self.state.apply_move(move, self.current_player)
        winner = self._check_winner()
        done = winner is not None or self._is_draw()

        reward = 0
        if winner is not None:
            reward = 1 if winner == self.current_player else -1
        elif done:
            reward = 0

        self.current_player *= -1  # Switch player

        return self.state, reward, done

    def _check_winner(self) -> Optional[int]:
        """Check if there is a winner.

        Returns:
            Player ID (1 or -1) if winner exists, None otherwise.
        """
        board = self.state.board

        # Check rows
        for row in range(3):
            if board[row, 0] != 0 and board[row, 0] == board[row, 1] == board[row, 2]:
                return board[row, 0]

        # Check columns
        for col in range(3):
            if board[0, col] != 0 and board[0, col] == board[1, col] == board[2, col]:
                return board[0, col]

        # Check diagonals
        if board[0, 0] != 0 and board[0, 0] == board[1, 1] == board[2, 2]:
            return board[0, 0]
        if board[0, 2] != 0 and board[0, 2] == board[1, 1] == board[2, 0]:
            return board[0, 2]

        return None

    def _is_draw(self) -> bool:
        """Check if game is a draw (board full, no winner).

        Returns:
            True if draw, False otherwise.
        """
        if self._check_winner() is not None:
            return False
        return len(self.state.get_valid_moves()) == 0

    def is_terminal(self) -> bool:
        """Check if current state is terminal (win or draw).

        Returns:
            True if game is over, False otherwise.
        """
        return self._check_winner() is not None or self._is_draw()

    def get_winner(self) -> Optional[int]:
        """Get winner if game is over.

        Returns:
            Player ID (1 or -1) if winner exists, None otherwise.
        """
        return self._check_winner()

    def get_current_player(self) -> int:
        """Get current player ID.

        Returns:
            1 for X, -1 for O.
        """
        return self.current_player

