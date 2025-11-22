# Tic-Tac-Toe AI Lab

An educational project exploring different game-playing algorithms for Tic-Tac-Toe. This repository implements multiple AI agents, from simple rule-based strategies to reinforcement learning, and provides an interactive Streamlit interface for playing, learning, and experimenting.

**Live Demo:** [tic-tac-toe-ailabs.streamlit.app](https://tic-tac-toe-ailabs.streamlit.app/)

**GitHub Repository:** [github.com/rohanrgupte/tic-tac-toe](https://github.com/rohanrgupte/tic-tac-toe)

**Created by:** [Rohan Gupte](https://github.com/rohanrgupte)

## Features

### AI Agents

- **Random Agent**: Baseline agent that selects moves uniformly at random
- **Rule-Based Agent**: Uses simple heuristics (win, block, prefer center/corners)
- **Minimax Agent**: Perfect play using minimax algorithm with alpha-beta pruning
- **MCTS Agent**: Monte Carlo Tree Search with configurable simulation count
- **Q-Learning Agent**: Tabular Q-learning with epsilon-greedy exploration

### Self-Play and Evaluation

- Run head-to-head comparisons between any two agents
- Train Q-learning agents through self-play
- Track win rates, draw rates, and training metrics
- Visualize training progress with reward curves

### Streamlit Interface

- **Play**: Play against AI agents or watch them play each other
- **Algorithms**: Learn how each algorithm works with explanations and equations
- **Experiments**: Compare agents or train Q-learning models

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rohanrgupte/tic-tac-toe.git
cd tic-tac-toe
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run app/main.py
```

The app will open in your default web browser. Use the sidebar to navigate between pages.

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

Run linting:
```bash
ruff check .
```

Format code:
```bash
black .
```

Set up pre-commit hooks:
```bash
pre-commit install
```

## Technical Notes

### Game Environment

The game is implemented as a deterministic environment with:
- 3x3 board representation (0 for empty, 1 for X, -1 for O)
- Clear state transitions and move validation
- Terminal state detection (win, draw, ongoing)

### Algorithms

#### Minimax

Minimax recursively evaluates all possible game outcomes, assuming optimal play from both players. With alpha-beta pruning, branches that cannot improve the best move are skipped, making the search more efficient.

The minimax recurrence:
```
minimax(s, player) = {
  utility(s) if terminal,
  max_a minimax(result(s, a), -player) if player = max,
  min_a minimax(result(s, a), -player) if player = min
}
```

#### MCTS

Monte Carlo Tree Search builds a search tree through repeated simulations:
1. **Selection**: Traverse tree using UCB1 to balance exploration/exploitation
2. **Expansion**: Add new child node
3. **Simulation**: Play random game to terminal state
4. **Backpropagation**: Update node statistics

UCB1 formula: `w_i/n_i + c * sqrt(ln(N)/n_i)`

#### Q-Learning

Q-learning learns a Q-table mapping (state, action) pairs to expected future rewards. The update rule:

```
Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') − Q(s, a)]
```

where α is the learning rate, γ is the discount factor, and r is the immediate reward.

Training is done through self-play against an opponent (typically a random agent initially). The agent uses epsilon-greedy exploration, gradually shifting from exploration to exploitation as epsilon decays.

### Q-Learning Training

The Q-learning agent can be trained through self-play:
1. Agent plays many games against an opponent
2. After each move, Q-values are updated using the Q-learning rule
3. Epsilon decays over time to reduce exploration
4. Trained Q-table can be saved and loaded for later use

With sufficient training, the agent learns near-optimal play for Tic-Tac-Toe's small state space.

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # Streamlit entry point
│   ├── game/
│   │   ├── environment.py   # Game rules and state transitions
│   │   └── state.py         # Board representation
│   ├── agents/
│   │   ├── random_agent.py
│   │   ├── rule_based_agent.py
│   │   ├── minimax_agent.py
│   │   ├── mcts_agent.py
│   │   └── q_learning_agent.py
│   ├── evaluation/
│   │   ├── self_play.py      # Self-play and training
│   │   └── metrics.py        # Evaluation metrics
│   └── ui/
│       ├── layout.py         # Page layouts
│       └── components.py     # UI components
├── tests/                    # Unit tests
├── requirements.txt
├── pyproject.toml           # Tool configuration
├── .pre-commit-config.yaml  # Pre-commit hooks
└── README.md
```

## Learning Objectives

This project demonstrates:
- Basic game environments and state representation
- Classical search algorithms (minimax, MCTS)
- Tabular reinforcement learning (Q-learning)
- Agent evaluation and self-play experiments
- Interactive visualization and experimentation

## Further Extensions

Ideas for extending this project:
- Implement deep Q-learning (DQN) for larger state spaces
- Add neural network evaluation to MCTS
- Implement other RL algorithms (SARSA, policy gradient methods)
- Extend to larger board sizes or different game rules
- Add tournament-style evaluation with ELO ratings
- Implement multi-agent reinforcement learning

## License

MIT License - see LICENSE file for details.

