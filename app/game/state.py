"""Board state representation and helper functions."""

from typing import List, Optional, Tuple

import numpy as np


class GameState:
    """Represents the current state of a Tic-Tac-Toe board.

    Board is represented as a 3x3 numpy array:
    - 0: empty cell
    - 1: player X
    - -1: player O
    """

    def __init__(self, board: Optional[np.ndarray] = None):
        """Initialize game state.

        Args:
            board: Optional 3x3 numpy array. If None, creates empty board.
        """
        if board is None:
            self.board = np.zeros((3, 3), dtype=int)
        else:
            self.board = board.copy()

    def __eq__(self, other) -> bool:
        """Check if two states are equal."""
        if not isinstance(other, GameState):
            return False
        return np.array_equal(self.board, other.board)

    def __hash__(self) -> int:
        """Hash state for use in dictionaries."""
        return hash(self.board.tobytes())

    def copy(self) -> "GameState":
        """Create a deep copy of the state."""
        return GameState(self.board)

    def get_valid_moves(self) -> List[int]:
        """Get list of valid move indices (0-8, row-major order).

        Returns:
            List of indices where board is empty.
        """
        flat = self.board.flatten()
        return [i for i in range(9) if flat[i] == 0]

    def is_valid_move(self, move: int) -> bool:
        """Check if a move is valid.

        Args:
            move: Move index (0-8).

        Returns:
            True if move is valid, False otherwise.
        """
        if move < 0 or move >= 9:
            return False
        flat = self.board.flatten()
        return flat[move] == 0

    def apply_move(self, move: int, player: int) -> "GameState":
        """Apply a move and return new state.

        Args:
            move: Move index (0-8).
            player: Player ID (1 for X, -1 for O).

        Returns:
            New GameState with move applied.

        Raises:
            ValueError: If move is invalid.
        """
        if not self.is_valid_move(move):
            raise ValueError(f"Invalid move: {move}")

        new_state = self.copy()
        row, col = divmod(move, 3)
        new_state.board[row, col] = player
        return new_state

    def get_cell(self, row: int, col: int) -> int:
        """Get cell value at position.

        Args:
            row: Row index (0-2).
            col: Column index (0-2).

        Returns:
            Cell value (0, 1, or -1).
        """
        return self.board[row, col]

    def to_string(self) -> str:
        """Convert board to human-readable string."""
        symbols = {0: " ", 1: "X", -1: "O"}
        lines = []
        for row in range(3):
            line = " | ".join(symbols[self.board[row, col]] for col in range(3))
            lines.append(line)
            if row < 2:
                lines.append("-" * 9)
        return "\n".join(lines)

    def encode(self) -> str:
        """Encode state as string for Q-learning.

        Returns:
            String representation of board state.
        """
        return "".join(str(x) for x in self.board.flatten())

