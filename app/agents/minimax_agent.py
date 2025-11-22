"""Minimax agent with alpha-beta pruning for perfect play."""

from typing import List, Optional, Tuple

from app.game.state import GameState


class MinimaxAgent:
    """Agent using minimax algorithm with alpha-beta pruning.

    Minimax recursively evaluates all possible game states, assuming
    both players play optimally. The algorithm maximizes the current
    player's score while minimizing the opponent's score.

    With alpha-beta pruning, we skip branches that cannot improve
    the best move found so far.
    """

    def __init__(self, depth: int = 9, use_alpha_beta: bool = True):
        """Initialize minimax agent.

        Args:
            depth: Maximum search depth. Default 9 (full game tree).
            use_alpha_beta: Whether to use alpha-beta pruning.
        """
        self.name = "Minimax"
        self.depth = depth
        self.use_alpha_beta = use_alpha_beta
        self._original_player = 1  # Default to X's perspective

    def select_action(self, state: GameState, player_id: int) -> int:
        """Select best move using minimax.

        Args:
            state: Current game state.
            player_id: Player ID (1 for X, -1 for O).

        Returns:
            Best move index (0-8).
        """
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")

        # Store original player for consistent evaluation perspective
        self._original_player = player_id

        if self.use_alpha_beta:
            best_move = None
            best_value = float("-inf")
            alpha = float("-inf")
            beta = float("inf")

            for move in valid_moves:
                next_state = state.apply_move(move, player_id)
                value = self._minimax_ab(
                    next_state, -player_id, self.depth - 1, alpha, beta
                )
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
        else:
            best_move = None
            best_value = float("-inf")

            for move in valid_moves:
                next_state = state.apply_move(move, player_id)
                value = self._minimax(next_state, -player_id, self.depth - 1)
                if value > best_value:
                    best_value = value
                    best_move = move

        return best_move

    def _minimax(
        self, state: GameState, player_id: int, depth: int
    ) -> float:
        """Minimax algorithm without pruning.

        Args:
            state: Current game state.
            player_id: Player whose turn it is.
            depth: Remaining search depth.

        Returns:
            Evaluation score from original player's perspective.
        """
        winner = self._check_winner(state)
        if winner is not None:
            # Return value from original player's perspective
            if winner == self._original_player:
                return 1  # Original player wins
            else:
                return -1  # Original player loses
        if self._is_draw(state):
            return 0
        if depth == 0:
            return 0  # Heuristic: assume draw at depth limit

        valid_moves = state.get_valid_moves()
        if player_id == self._original_player:  # Maximizing player (original player's turn)
            max_value = float("-inf")
            for move in valid_moves:
                next_state = state.apply_move(move, player_id)
                value = self._minimax(next_state, -player_id, depth - 1)
                max_value = max(max_value, value)
            return max_value
        else:  # Minimizing player (opponent's turn)
            min_value = float("inf")
            for move in valid_moves:
                next_state = state.apply_move(move, player_id)
                value = self._minimax(next_state, -player_id, depth - 1)
                min_value = min(min_value, value)
            return min_value

    def _minimax_ab(
        self,
        state: GameState,
        player_id: int,
        depth: int,
        alpha: float,
        beta: float,
    ) -> float:
        """Minimax with alpha-beta pruning.

        Args:
            state: Current game state.
            player_id: Player whose turn it is.
            depth: Remaining search depth.
            alpha: Best value for maximizing player.
            beta: Best value for minimizing player.

        Returns:
            Evaluation score from original player's perspective.
        """
        winner = self._check_winner(state)
        if winner is not None:
            # Return value from original player's perspective
            if winner == self._original_player:
                return 1  # Original player wins
            else:
                return -1  # Original player loses
        if self._is_draw(state):
            return 0
        if depth == 0:
            return 0

        valid_moves = state.get_valid_moves()
        if player_id == self._original_player:  # Maximizing player (original player's turn)
            max_value = float("-inf")
            for move in valid_moves:
                next_state = state.apply_move(move, player_id)
                value = self._minimax_ab(
                    next_state, -player_id, depth - 1, alpha, beta
                )
                max_value = max(max_value, value)
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break  # Alpha-beta cutoff
            return max_value
        else:  # Minimizing player (opponent's turn)
            min_value = float("inf")
            for move in valid_moves:
                next_state = state.apply_move(move, player_id)
                value = self._minimax_ab(
                    next_state, -player_id, depth - 1, alpha, beta
                )
                min_value = min(min_value, value)
                beta = min(beta, min_value)
                if beta <= alpha:
                    break  # Alpha-beta cutoff
            return min_value

    def _check_winner(self, state: GameState) -> Optional[int]:
        """Check if there is a winner.

        Returns:
            Player ID (1 or -1) if winner exists, None otherwise.
        """
        board = state.board

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

    def _is_draw(self, state: GameState) -> bool:
        """Check if game is a draw.

        Returns:
            True if draw, False otherwise.
        """
        if self._check_winner(state) is not None:
            return False
        return len(state.get_valid_moves()) == 0

