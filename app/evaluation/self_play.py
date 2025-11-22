"""Self-play functions for training and evaluation."""

from typing import List, Tuple

from app.agents.q_learning_agent import QLearningAgent
from app.agents.random_agent import RandomAgent
from app.evaluation.metrics import GameMetrics, evaluate_agents
from app.game.environment import GameEnvironment
from app.game.state import GameState


def run_self_play(agent_x, agent_o, num_games: int = 1000) -> GameMetrics:
    """Run self-play between two agents.

    Args:
        agent_x: Agent playing as X.
        agent_o: Agent playing as O.
        num_games: Number of games to play.

    Returns:
        GameMetrics with results.
    """
    metrics, _ = evaluate_agents(agent_x, agent_o, num_games)
    return metrics


def train_q_learning(
    agent: QLearningAgent,
    opponent: RandomAgent,
    num_episodes: int = 1000,
    verbose: bool = True,
) -> List[dict]:
    """Train Q-learning agent through self-play.

    Args:
        agent: Q-learning agent to train.
        opponent: Opponent agent (typically RandomAgent).
        num_episodes: Number of training episodes.
        verbose: Whether to print progress.

    Returns:
        List of training metrics per episode.
    """
    training_history = []

    for episode in range(num_episodes):
        env = GameEnvironment()
        # Random first player for better learning
        first_player = 1 if episode % 2 == 0 else -1
        state = env.reset()
        if first_player == -1:
            env.current_player = -1
        
        episode_reward = 0
        episode_steps = 0

        # Determine which player the agent is
        agent_player = first_player
        opponent_player = -first_player

        while not env.is_terminal():
            current_player = env.get_current_player()
            if current_player == agent_player:
                action = agent.select_action(state, agent_player)
                next_state, reward, done = env.step(action)
                agent.update(
                    state, action, reward, next_state, agent_player, done
                )
                episode_reward += reward if agent_player == 1 else -reward
            else:
                action = opponent.select_action(state, opponent_player)
                next_state, reward, done = env.step(action)
                episode_reward += reward if agent_player == 1 else -reward

            state = next_state
            episode_steps += 1

            if done:
                break

        # Finalize game for backward propagation
        winner = env.get_winner()
        agent.finalize_game(winner)
        agent.decay_epsilon()

        if verbose and (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Epsilon: {agent.epsilon:.3f}")

        training_history.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "steps": episode_steps,
                "epsilon": agent.epsilon,
            }
        )

    return training_history

