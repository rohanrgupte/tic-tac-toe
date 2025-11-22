"""UI components for rendering game board and controls."""

from typing import Callable, Optional

import streamlit as st

from app.game.state import GameState


def render_board(
    state: GameState,
    on_cell_click: Optional[Callable[[int], None]] = None,
    disabled: bool = False,
) -> Optional[int]:
    """Render Tic-Tac-Toe board with clickable cells.

    Args:
        state: Current game state.
        on_cell_click: Callback function when cell is clicked (deprecated, use return value).
        disabled: Whether board is disabled.

    Returns:
        Selected move index if clicked, None otherwise.
    """
    selected_move = None
    symbols = {0: "", 1: "X", -1: "O"}

    cols = st.columns(3)
    for row in range(3):
        for col in range(3):
            cell_value = state.get_cell(row, col)
            move_idx = row * 3 + col
            symbol = symbols[cell_value]

            with cols[col]:
                if disabled or cell_value != 0:
                    st.button(
                        symbol if symbol else " ",
                        key=f"cell_{move_idx}",
                        disabled=True,
                        use_container_width=True,
                    )
                else:
                    if st.button(
                        " ",
                        key=f"cell_{move_idx}",
                        disabled=disabled,
                        use_container_width=True,
                    ):
                        selected_move = move_idx
                        if on_cell_click:
                            on_cell_click(move_idx)

    return selected_move


def render_game_status(
    state: GameState,
    current_player: int,
    winner: Optional[int],
    is_draw: bool,
):
    """Render current game status.

    Args:
        state: Current game state.
        current_player: Current player ID (1 for X, -1 for O).
        winner: Winner ID if game is over, None otherwise.
        is_draw: Whether game is a draw.
    """
    if winner is not None:
        winner_name = "X" if winner == 1 else "O"
        st.success(f"Game Over: {winner_name} wins!")
    elif is_draw:
        st.info("Game Over: Draw!")
    else:
        player_name = "X" if current_player == 1 else "O"
        st.write(f"Current Player: **{player_name}**")


def render_agent_selector(key_prefix: str, label: str = "Agent"):
    """Render agent selection dropdown.

    Args:
        key_prefix: Unique key prefix for Streamlit state.
        label: Label for the selector.

    Returns:
        Selected agent name.
    """
    agents = [
        "Random",
        "Rule-Based",
        "Minimax",
        "MCTS",
        "Q-Learning",
    ]
    return st.selectbox(label, agents, key=f"{key_prefix}_agent")

