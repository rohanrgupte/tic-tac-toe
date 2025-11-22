"""Tests for game environment."""

import pytest

from app.game.environment import GameEnvironment
from app.game.state import GameState


def test_reset():
    """Test environment reset."""
    env = GameEnvironment()
    state = env.reset()
    assert len(state.get_valid_moves()) == 9
    assert env.get_current_player() == 1


def test_win_detection_row():
    """Test win detection in a row."""
    env = GameEnvironment()
    env.reset()

    # X wins in first row
    env.step(0)  # X
    env.step(3)  # O
    env.step(1)  # X
    env.step(4)  # O
    state, reward, done = env.step(2)  # X wins

    assert done is True
    assert reward == 1
    assert env.get_winner() == 1


def test_win_detection_column():
    """Test win detection in a column."""
    env = GameEnvironment()
    env.reset()

    # O wins in first column
    env.step(1)  # X
    env.step(0)  # O
    env.step(2)  # X
    env.step(3)  # O
    env.step(4)  # X
    state, reward, done = env.step(6)  # O wins

    assert done is True
    assert reward == -1
    assert env.get_winner() == -1


def test_win_detection_diagonal():
    """Test win detection in diagonal."""
    env = GameEnvironment()
    env.reset()

    # X wins in main diagonal
    env.step(0)  # X
    env.step(1)  # O
    env.step(4)  # X
    env.step(2)  # O
    state, reward, done = env.step(8)  # X wins

    assert done is True
    assert reward == 1
    assert env.get_winner() == 1


def test_draw_detection():
    """Test draw detection."""
    env = GameEnvironment()
    env.reset()

    # Create a draw
    moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]
    for move in moves[:-1]:
        env.step(move)

    state, reward, done = env.step(moves[-1])

    assert done is True
    assert reward == 0
    assert env.get_winner() is None


def test_invalid_move():
    """Test invalid move raises error."""
    env = GameEnvironment()
    env.reset()
    env.step(0)

    with pytest.raises(ValueError):
        env.step(0)  # Cell already occupied


def test_player_switching():
    """Test player switching."""
    env = GameEnvironment()
    env.reset()
    assert env.get_current_player() == 1

    env.step(0)
    assert env.get_current_player() == -1

    env.step(1)
    assert env.get_current_player() == 1


def test_terminal_state():
    """Test terminal state detection."""
    env = GameEnvironment()
    env.reset()
    assert env.is_terminal() is False

    # Create a win
    env.step(0)
    env.step(3)
    env.step(1)
    env.step(4)
    env.step(2)

    assert env.is_terminal() is True

