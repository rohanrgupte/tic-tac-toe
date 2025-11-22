"""Tests for AI agents."""

import pytest

from app.agents.minimax_agent import MinimaxAgent
from app.agents.mcts_agent import MCTSAgent
from app.agents.q_learning_agent import QLearningAgent
from app.agents.random_agent import RandomAgent
from app.agents.rule_based_agent import RuleBasedAgent
from app.game.state import GameState


def test_random_agent():
    """Test random agent produces valid moves."""
    agent = RandomAgent()
    state = GameState()

    for _ in range(10):
        move = agent.select_action(state, 1)
        assert move in state.get_valid_moves()
        state = state.apply_move(move, 1)
        if len(state.get_valid_moves()) == 0:
            break


def test_rule_based_agent_winning_move():
    """Test rule-based agent takes winning move."""
    agent = RuleBasedAgent()
    # Create state where X can win
    state = GameState()
    state = state.apply_move(0, 1)  # X
    state = state.apply_move(3, -1)  # O
    state = state.apply_move(1, 1)  # X

    move = agent.select_action(state, 1)
    assert move == 2  # Should take winning move


def test_rule_based_agent_block():
    """Test rule-based agent blocks opponent."""
    agent = RuleBasedAgent()
    # Create state where O can win
    state = GameState()
    state = state.apply_move(0, 1)  # X
    state = state.apply_move(3, -1)  # O
    state = state.apply_move(1, 1)  # X
    state = state.apply_move(4, -1)  # O

    move = agent.select_action(state, 1)
    assert move == 5  # Should block O's winning move


def test_minimax_agent():
    """Test minimax agent produces valid moves."""
    agent = MinimaxAgent()
    state = GameState()

    move = agent.select_action(state, 1)
    assert move in state.get_valid_moves()


def test_minimax_agent_perfect_play():
    """Test minimax agent plays optimally."""
    agent = MinimaxAgent()
    state = GameState()

    # From empty board, minimax should prefer center or corner
    move = agent.select_action(state, 1)
    assert move in [0, 2, 4, 6, 8]  # Center or corners


def test_mcts_agent():
    """Test MCTS agent produces valid moves."""
    agent = MCTSAgent(num_simulations=100)
    state = GameState()

    move = agent.select_action(state, 1)
    assert move in state.get_valid_moves()


def test_q_learning_agent():
    """Test Q-learning agent produces valid moves."""
    agent = QLearningAgent()
    state = GameState()

    move = agent.select_action(state, 1)
    assert move in state.get_valid_moves()


def test_q_learning_update():
    """Test Q-learning update mechanism."""
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9)
    state1 = GameState()
    state2 = state1.apply_move(0, 1)

    # Update Q-value
    agent.update(state1, 0, 1.0, state2, 1, False)

    # Check Q-table was updated
    state_key = agent._normalize_state(state1, 1)
    assert state_key in agent.q_table


def test_q_learning_epsilon_decay():
    """Test epsilon decay."""
    agent = QLearningAgent(epsilon=0.5, epsilon_decay=0.9)
    initial_epsilon = agent.epsilon

    agent.decay_epsilon()
    assert agent.epsilon < initial_epsilon
    assert agent.epsilon == 0.5 * 0.9


def test_q_learning_save_load():
    """Test Q-learning save and load."""
    import os
    import tempfile

    agent1 = QLearningAgent()
    state = GameState()
    agent1.update(state, 0, 1.0, state.apply_move(0, 1), 1, False)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        agent1.save(temp_path)
        agent2 = QLearningAgent()
        agent2.load(temp_path)

        assert agent1.q_table == agent2.q_table
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_agent_no_valid_moves():
    """Test agents handle terminal states."""
    agent = RandomAgent()
    state = GameState()

    # Fill board
    for i in range(9):
        player = 1 if i % 2 == 0 else -1
        if i < 8:
            state = state.apply_move(i, player)

    # No valid moves left
    with pytest.raises(ValueError):
        agent.select_action(state, 1)

