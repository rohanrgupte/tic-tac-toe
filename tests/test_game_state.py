"""Tests for game state representation."""

import numpy as np
import pytest

from app.game.state import GameState


def test_empty_state():
    """Test initial empty state."""
    state = GameState()
    assert len(state.get_valid_moves()) == 9
    assert all(state.board.flatten() == 0)


def test_apply_move():
    """Test applying a move."""
    state = GameState()
    new_state = state.apply_move(0, 1)
    assert new_state.get_cell(0, 0) == 1
    assert state.get_cell(0, 0) == 0  # Original unchanged


def test_invalid_move():
    """Test invalid move raises error."""
    state = GameState()
    state = state.apply_move(0, 1)
    with pytest.raises(ValueError):
        state.apply_move(0, -1)  # Cell already occupied


def test_get_valid_moves():
    """Test getting valid moves."""
    state = GameState()
    assert len(state.get_valid_moves()) == 9

    state = state.apply_move(0, 1)
    assert len(state.get_valid_moves()) == 8
    assert 0 not in state.get_valid_moves()


def test_is_valid_move():
    """Test move validation."""
    state = GameState()
    assert state.is_valid_move(0) is True
    assert state.is_valid_move(9) is False  # Out of bounds

    state = state.apply_move(0, 1)
    assert state.is_valid_move(0) is False  # Already occupied


def test_state_equality():
    """Test state equality."""
    state1 = GameState()
    state2 = GameState()
    assert state1 == state2

    state1 = state1.apply_move(0, 1)
    assert state1 != state2


def test_state_copy():
    """Test state copying."""
    state1 = GameState()
    state1 = state1.apply_move(0, 1)
    state2 = state1.copy()

    assert state1 == state2
    assert state1 is not state2

    state2 = state2.apply_move(1, -1)
    assert state1 != state2


def test_encode():
    """Test state encoding."""
    state = GameState()
    encoded = state.encode()
    assert len(encoded) == 9
    assert all(c == "0" for c in encoded)

    state = state.apply_move(0, 1)
    encoded = state.encode()
    assert encoded[0] == "1"

