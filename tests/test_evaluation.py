"""Tests for evaluation and self-play."""

from app.agents.random_agent import RandomAgent
from app.evaluation.metrics import GameMetrics, evaluate_agents


def test_game_metrics():
    """Test game metrics tracking."""
    metrics = GameMetrics()

    metrics.record_result(1)  # X wins
    metrics.record_result(-1)  # O wins
    metrics.record_result(None)  # Draw

    assert metrics.total_games == 3
    assert metrics.wins_x == 1
    assert metrics.wins_o == 1
    assert metrics.draws == 1
    assert metrics.get_win_rate_x() == 1 / 3
    assert metrics.get_win_rate_o() == 1 / 3
    assert metrics.get_draw_rate() == 1 / 3


def test_evaluate_agents():
    """Test agent evaluation."""
    agent_x = RandomAgent()
    agent_o = RandomAgent()

    metrics, history = evaluate_agents(agent_x, agent_o, num_games=10)

    assert metrics.total_games == 10
    assert len(history) == 10
    assert metrics.wins_x + metrics.wins_o + metrics.draws == 10


def test_evaluate_agents_all_games_complete():
    """Test that all evaluated games complete."""
    agent_x = RandomAgent()
    agent_o = RandomAgent()

    metrics, history = evaluate_agents(agent_x, agent_o, num_games=5)

    for game in history:
        assert "winner" in game
        assert game["winner"] in [1, -1, None]

