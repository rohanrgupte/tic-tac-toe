"""Metrics and evaluation functions for comparing agents."""

from typing import Dict, List, Optional, Tuple

from app.agents.random_agent import RandomAgent
from app.game.environment import GameEnvironment
from app.game.state import GameState


class GameMetrics:
    """Track game statistics."""

    def __init__(self):
        """Initialize empty metrics."""
        self.wins_x = 0
        self.wins_o = 0
        self.draws = 0
        self.total_games = 0

    def record_result(self, winner: Optional[int]):
        """Record game result.

        Args:
            winner: Winner ID (1 for X, -1 for O, None for draw).
        """
        self.total_games += 1
        if winner == 1:
            self.wins_x += 1
        elif winner == -1:
            self.wins_o += 1
        else:
            self.draws += 1

    def get_win_rate_x(self) -> float:
        """Get win rate for player X.

        Returns:
            Win rate as fraction (0.0 to 1.0).
        """
        if self.total_games == 0:
            return 0.0
        return self.wins_x / self.total_games

    def get_win_rate_o(self) -> float:
        """Get win rate for player O.

        Returns:
            Win rate as fraction (0.0 to 1.0).
        """
        if self.total_games == 0:
            return 0.0
        return self.wins_o / self.total_games

    def get_draw_rate(self) -> float:
        """Get draw rate.

        Returns:
            Draw rate as fraction (0.0 to 1.0).
        """
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary.

        Returns:
            Dictionary with win rates and draw rate.
        """
        return {
            "wins_x": self.wins_x,
            "wins_o": self.wins_o,
            "draws": self.draws,
            "total_games": self.total_games,
            "win_rate_x": self.get_win_rate_x(),
            "win_rate_o": self.get_win_rate_o(),
            "draw_rate": self.get_draw_rate(),
        }


def evaluate_agents(
    agent_x, agent_o, num_games: int = 1000
) -> Tuple[GameMetrics, List[Dict]]:
    """Evaluate two agents by playing multiple games.

    Args:
        agent_x: Agent playing as X.
        agent_o: Agent playing as O.
        num_games: Number of games to play.

    Returns:
        Tuple of (metrics, game_history).
    """
    metrics = GameMetrics()
    game_history = []

    for game_num in range(num_games):
        env = GameEnvironment()
        state = env.reset()
        history = []

        while not env.is_terminal():
            current_player = env.get_current_player()
            if current_player == 1:
                move = agent_x.select_action(state, 1)
            else:
                move = agent_o.select_action(state, -1)

            state, reward, done = env.step(move)
            history.append({"move": move, "player": current_player, "reward": reward})

            if done:
                break

        winner = env.get_winner()
        metrics.record_result(winner)
        game_history.append({"winner": winner, "history": history})

    return metrics, game_history

