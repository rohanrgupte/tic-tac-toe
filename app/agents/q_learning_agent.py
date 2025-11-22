"""Q-learning agent - learns by playing games and updating Q-values."""

import json
import os
import random
from typing import Dict, List, Optional

from app.game.state import GameState


class QLearningAgent:
    """Agent using tabular Q-learning.

    Q-learning learns a Q-table mapping (state, action) pairs to expected
    future rewards. The Q-value is updated using:

    Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') − Q(s, a)]

    where:
    - α (alpha) is the learning rate
    - γ (gamma) is the discount factor
    - r is the immediate reward
    - s' is the next state
    """

    def __init__(
        self,
        learning_rate: float = 0.1,  # How fast to learn (α in formula)
        discount_factor: float = 0.9,  # How much to value future rewards (γ)
        epsilon: float = 0.1,  # Exploration rate (chance to try random moves)
        epsilon_decay: float = 0.995,  # How fast exploration decreases
        min_epsilon: float = 0.01,  # Minimum exploration rate
    ):
        self.name = "Q-Learning"
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self._initial_epsilon = epsilon  # For reset
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Q-table: maps state (as string) -> action (0-8) -> Q-value (float)
        self.q_table: Dict[str, Dict[int, float]] = {}
        
        # Tracking for UI display
        self.track_history = False
        self.game_history: List[Dict] = []
        self.current_game_steps: List[Dict] = []
        self.total_games_played = 0

    def select_action(self, state: GameState, player_id: int) -> int:
        """Choose a move using epsilon-greedy policy.
        
        Epsilon-greedy means:
        - With probability epsilon: explore (random move)
        - Otherwise: exploit (best move from Q-table)
        """
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")

        # Normalize state to canonical form (X's perspective)
        state_key = self._normalize_state(state, player_id)
        q_values = self._get_q_values(state_key)

        # Explore or exploit?
        is_exploration = random.random() < self.epsilon
        
        if is_exploration:
            # Explore: try random move
            selected_move = random.choice(valid_moves)
        else:
            # Exploit: choose move with highest Q-value
            best_value = float("-inf")
            best_moves = []

            for move in valid_moves:
                value = q_values.get(move, 0.0)
                if value > best_value:
                    best_value = value
                    best_moves = [move]
                elif value == best_value:
                    best_moves.append(move)

            selected_move = random.choice(best_moves) if best_moves else random.choice(valid_moves)

        # Track move for backward propagation (updating Q-values at game end)
        step_info = {
            "state": state.copy(),
            "state_key": state_key,
            "player_id": player_id,
            "action": selected_move,
            "is_exploration": is_exploration,
            "q_values_before": {k: v for k, v in q_values.items()} if self.track_history else {},
            "valid_moves": valid_moves,
        }
        self.current_game_steps.append(step_info)

        return selected_move

    def update(
        self,
        state: GameState,
        action: int,
        reward: float,
        next_state: GameState,
        player_id: int,
        done: bool,
    ):
        """Update Q-value using Q-learning formula.
        
        Formula: Q(s,a) = Q(s,a) + α[r + γ*max Q(s',a') - Q(s,a)]
        where:
        - s = current state, a = action taken
        - r = reward (1 if win, -1 if lose, 0 otherwise)
        - s' = next state
        - α = learning rate, γ = discount factor
        """
        state_key = self._normalize_state(state, player_id)

        q_values = self._get_q_values(state_key)
        current_q = q_values.get(action, 0.0)

        # Calculate target Q-value
        if done:
            # Game ended: target is just the reward
            target_q = reward
        else:
            # Game continues: target is reward + discounted value of next state
            next_state_key = self._normalize_state(next_state, player_id)
            next_q_values = self._get_q_values(next_state_key)
            next_valid_moves = next_state.get_valid_moves()
            
            if next_q_values and next_valid_moves:
                # Get best Q-value for next state
                max_next_q = max(
                    next_q_values.get(move, 0.0) for move in next_valid_moves
                )
            else:
                max_next_q = 0.0
            
            target_q = reward + self.discount_factor * max_next_q

        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        q_values[action] = new_q
        self.q_table[state_key] = q_values

        # Update tracking info for UI
        if self.track_history and self.current_game_steps:
            for step in reversed(self.current_game_steps):
                if step.get("state_key") == state_key and step.get("action") == action:
                    step["reward"] = reward
                    step["next_state"] = next_state
                    step["q_value_before"] = current_q
                    step["q_value_after"] = new_q
                    step["target_q"] = target_q
                    step["q_values_after"] = {k: v for k, v in q_values.items()}
                    break

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save Q-table to file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        save_data = {
            "q_table": self.q_table,
            "total_games_played": self.total_games_played,
            "epsilon": self.epsilon,
        }
        with open(filepath, "w") as f:
            json.dump(save_data, f)

    def load(self, filepath: str):
        """Load Q-table from file."""
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "q_table" in data:
                    self.q_table = data["q_table"]
                    self.total_games_played = data.get("total_games_played", 0)
                    self.epsilon = data.get("epsilon", self.epsilon)
                else:
                    self.q_table = data
                # Convert string keys to ints
                for state_key in self.q_table:
                    self.q_table[state_key] = {
                        int(k): v for k, v in self.q_table[state_key].items()
                    }

    def _normalize_state(self, state: GameState, player_id: int) -> str:
        """Convert state to canonical form (always from X's perspective).
        
        This helps the agent learn faster by treating symmetric positions
        (e.g., X in corner vs O in corner) as the same.
        """
        board = state.board.copy()
        if player_id == -1:
            # Flip board: O's perspective becomes X's perspective
            board = -board
        return "".join(str(int(x)) for x in board.flatten())

    def reset_learning(self):
        """Clear all learning data and start fresh."""
        self.q_table = {}
        self.game_history = []
        self.current_game_steps = []
        self.total_games_played = 0
        self.epsilon = self._initial_epsilon if hasattr(self, '_initial_epsilon') else 0.1
        self.track_history = False

    def enable_history_tracking(self):
        """Enable detailed tracking for UI display."""
        self.track_history = True
        self.current_game_steps = []

    def disable_history_tracking(self):
        """Disable tracking."""
        if self.track_history and self.current_game_steps:
            self.game_history.append(self.current_game_steps.copy())
            if len(self.game_history) > 100:
                self.game_history.pop(0)
        self.track_history = False
        self.current_game_steps = []

    def finalize_game(self, winner: Optional[int]):
        """Update Q-values at game end using backward propagation.
        
        Backward propagation: update Q-values from the end of the game
        backward to the beginning. This is more effective than forward
        updates because we know the final outcome.
        """
        self.total_games_played += 1
        
        if self.current_game_steps:
            # Get this agent's player ID
            agent_player = self.current_game_steps[0].get("player_id", 1) if self.current_game_steps else 1
            
            # Calculate final reward
            if winner is None:
                final_reward = 0.0  # Draw
            elif winner == agent_player:
                final_reward = 1.0  # Won
            else:
                final_reward = -1.0  # Lost
            
            # Backward propagation: update from end to beginning
            future_value = final_reward  # Start with final reward
            
            for step in reversed(self.current_game_steps):
                step_player = step.get("player_id")
                step_state = step.get("state")
                step_action = step.get("action")
                
                # Only update this agent's moves
                if step_state and step_action is not None and step_player == agent_player:
                    state_key = self._normalize_state(step_state, step_player)
                    q_values = self._get_q_values(state_key)
                    current_q = q_values.get(step_action, 0.0)
                    
                    # Store the future_value that was USED for this step's calculation (BEFORE we update it)
                    future_value_used_for_this_step = future_value
                    
                    # Update Q-value: immediate reward (0) + discounted future value
                    target_q = 0.0 + self.discount_factor * future_value
                    new_q = current_q + self.learning_rate * (target_q - current_q)
                    q_values[step_action] = new_q
                    self.q_table[state_key] = q_values
                    
                    # Future value for next (earlier) step is this updated Q-value
                    future_value = new_q
                    
                    # Update tracking info
                    if self.track_history:
                        step["reward"] = final_reward if step == self.current_game_steps[-1] else 0.0
                        step["q_value_before"] = current_q
                        step["q_value_after"] = new_q
                        step["target_q"] = target_q
                        # Store the future_value that was USED for this step's calculation
                        step["future_value_used"] = future_value_used_for_this_step  # The value used to calculate target_q
                        step["future_value"] = new_q  # The value that will be used for next (earlier) step
                        step["q_values_after"] = {k: v for k, v in q_values.items()}
                        step["game_result"] = winner
                        step["is_backward_prop"] = True  # Mark as backward propagation update
        
        if self.track_history and self.current_game_steps:
            self.game_history.append(self.current_game_steps.copy())
            if len(self.game_history) > 100:
                self.game_history.pop(0)
        self.current_game_steps = []

    def get_last_game(self) -> Optional[List[Dict]]:
        """Get the last played game history."""
        if self.game_history:
            return self.game_history[-1]
        return None

    def get_recent_games(self, n: int = 25) -> List[Dict]:
        """Get results of last n games."""
        if not self.game_history:
            return []
        recent = self.game_history[-n:]
        results = []
        for game in recent:
            if game:
                result = game[0].get("game_result", None)
                results.append({
                    "winner": result,
                    "num_steps": len(game),
                })
        return results

    def _get_q_values(self, state_key: str) -> Dict[int, float]:
        """Get Q-values for a state (create if doesn't exist)."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        return self.q_table[state_key]
