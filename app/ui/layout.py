"""Main layout and page navigation for Streamlit app."""

import random
from typing import Dict
import streamlit as st

from app.ui import components


def render_play_page():
    """Render the main play page."""
    st.header("Play Tic-Tac-Toe")

    # Initialize session state
    if "game_state" not in st.session_state:
        from app.game.environment import GameEnvironment

        st.session_state.env = GameEnvironment()
        st.session_state.game_state = st.session_state.env.reset()
        st.session_state.game_over = False

    # Agent selection
    col1, col2 = st.columns(2)
    with col1:
        player_x_type = components.render_agent_selector("player_x", "Player X")
        is_human_x = st.checkbox("Human", key="human_x")
    with col2:
        player_o_type = components.render_agent_selector("player_o", "Player O")
        is_human_o = st.checkbox("Human", key="human_o")


    # Get agent instances
    agent_x = _get_agent(player_x_type, "x")
    agent_o = _get_agent(player_o_type, "o")

    # Q-Learning training section (before simulation)
    is_qlearning_selected = player_x_type == "Q-Learning" or player_o_type == "Q-Learning"
    if is_qlearning_selected:
        st.write("---")
        q_agent = agent_x if player_x_type == "Q-Learning" else agent_o
        
        st.subheader("Q-Learning Training")
        st.write("**Train Q-Learning:** Play games against a random opponent to learn optimal strategies. The agent will continue learning during simulations.")
        
        # Show current training status
        if hasattr(q_agent, "total_games_played"):
            total_games = q_agent.total_games_played
            q_table_size = len(q_agent.q_table) if hasattr(q_agent, "q_table") else 0
            st.info(f"**Current status:** {total_games} games learned from | Q-table size: {q_table_size} states | Epsilon: {q_agent.epsilon:.3f}")
        
        col1, col2 = st.columns(2)
        with col1:
            num_training_games = st.number_input(
                "Number of training games against Random",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100,
                key="qlearning_training_games",
                help="Q-learning will play this many games against a random opponent to learn before simulation."
            )
        with col2:
            if st.button("Train Q-Learning", key="train_qlearning"):
                from app.agents.random_agent import RandomAgent
                from app.evaluation.self_play import train_q_learning
                
                random_opponent = RandomAgent()
                
                # Enable history tracking if not already enabled
                if hasattr(q_agent, "enable_history_tracking") and not q_agent.track_history:
                    q_agent.enable_history_tracking()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Custom training with progress updates
                from app.game.environment import GameEnvironment
                import random as random_module
                
                training_history = []
                for episode in range(num_training_games):
                    env = GameEnvironment()
                    # Random first player for better learning
                    first_player = random_module.choice([1, -1])
                    state = env.reset()
                    if first_player == -1:
                        env.current_player = -1
                    
                    agent_player = first_player
                    opponent_player = -first_player
                    
                    while not env.is_terminal():
                        current_player = env.get_current_player()
                        if current_player == agent_player:
                            action = q_agent.select_action(state, agent_player)
                            next_state, reward, done = env.step(action)
                            q_agent.update(state, action, reward, next_state, agent_player, done)
                        else:
                            action = random_opponent.select_action(state, opponent_player)
                            next_state, reward, done = env.step(action)
                        
                        state = next_state
                        if done:
                            break
                    
                    # Finalize game for backward propagation
                    winner = env.get_winner()
                    q_agent.finalize_game(winner)
                    q_agent.decay_epsilon()
                    
                    # Update progress
                    if (episode + 1) % max(1, num_training_games // 20) == 0 or (episode + 1) == num_training_games:
                        progress = (episode + 1) / num_training_games
                        progress_bar.progress(progress)
                        status_text.text(f"Training: {episode + 1}/{num_training_games} games (Epsilon: {q_agent.epsilon:.3f})")
                
                # Save the trained model
                if hasattr(q_agent, "save"):
                    try:
                        q_agent.save("models/q_table.json")
                    except Exception:
                        pass
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"Training complete! Q-Learning agent has learned from {q_agent.total_games_played} total games.")
                st.rerun()
        
        # Reset learning button
        if st.button("Reset Learning Data", key="reset_qlearning_training", type="secondary"):
            if hasattr(q_agent, "reset_learning"):
                q_agent.reset_learning()
            # Delete saved model file
            import os
            model_path = "models/q_table.json"
            if os.path.exists(model_path):
                os.remove(model_path)
            st.success("Learning data reset! Q-table cleared.")
            st.rerun()

    # MCTS tree building (must happen before simulation)
    is_mcts_selected = player_x_type == "MCTS" or player_o_type == "MCTS"
    if is_mcts_selected:
        st.write("---")
        mcts_agent = agent_x if player_x_type == "MCTS" else agent_o
        
        st.subheader("MCTS Configuration")
        st.write("**How MCTS works:** For each move, MCTS runs multiple iterations of tree traversal, node expansion, rollout (simulation), and backpropagation. The number below determines how many iterations to run before making each move.")
        mcts_num_sim = st.number_input(
            "Number of MCTS iterations per move (Selection → Expansion → Rollout → Backpropagation)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            key="mcts_tree_simulations",
            help="Each iteration runs the full MCTS cycle: traverse tree, expand node, simulate random game, backpropagate results. More iterations = better move selection but slower."
        )
        
        # Update the agent's num_simulations to match the input
        mcts_agent.num_simulations = mcts_num_sim
        
        if st.button("Build & Visualize MCTS Tree", key="build_mcts_tree"):
            initial_state = st.session_state.game_state.copy()
            _simulate_mcts_for_visualization(mcts_agent, mcts_num_sim)
            st.session_state.mcts_tree_built = True
            st.session_state.mcts_initial_state = initial_state
            st.success(f"MCTS tree built successfully! Agent configured to use {mcts_num_sim} iterations per move.")
            st.rerun()
        
        # Show tree if built
        if st.session_state.get("mcts_tree_built", False):
            initial_state = st.session_state.get("mcts_initial_state")
            if initial_state is not None:
                with st.expander("View MCTS Tree", expanded=False):
                    _render_mcts_tree_visualization(mcts_agent, initial_state, mcts_num_sim)
        
        # Show current configuration
        st.info(f"**Current MCTS configuration:** {mcts_agent.num_simulations} iterations per move. This determines how thoroughly MCTS explores each position before choosing a move.")

    # Simulation feature for all agents (happens after tree building for MCTS)
    st.write("---")
    st.subheader("Simulation")
    st.write("Run games between the selected agents to see their performance.")
    
    num_sim_games = st.number_input("Number of games to simulate", min_value=10, max_value=10000, value=1000, step=10, key="num_sim_games")
    
    if st.button("Run Simulation", key="run_simulation"):
        _simulate_games(agent_x, agent_o, player_x_type, player_o_type, num_sim_games)
        st.rerun()
    
    # Show simulation results if they exist
    if "simulation_results" in st.session_state:
        st.write("---")
        st.subheader("Last Simulation Results")
        results = st.session_state.simulation_results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("X Wins", results.get("wins_x", 0), f"{results.get('win_rate_x', 0)*100:.1f}%")
        with col2:
            st.metric("O Wins", results.get("wins_o", 0), f"{results.get('win_rate_o', 0)*100:.1f}%")
        with col3:
            st.metric("Draws", results.get("draws", 0), f"{results.get('draw_rate', 0)*100:.1f}%")
        with col4:
            st.metric("Total Games", results.get("total_games", 0))
    
    # Q-Learning specific features (after simulation)
    is_qlearning_selected = player_x_type == "Q-Learning" or player_o_type == "Q-Learning"
    
    # Enable history tracking for Q-learning agents
    if player_x_type == "Q-Learning" and hasattr(agent_x, "enable_history_tracking"):
        if not agent_x.track_history:
            agent_x.enable_history_tracking()
    if player_o_type == "Q-Learning" and hasattr(agent_o, "enable_history_tracking"):
        if not agent_o.track_history:
            agent_o.enable_history_tracking()
    
    if is_qlearning_selected:
        st.write("---")
        q_agent = agent_x if player_x_type == "Q-Learning" else agent_o
        
        # Show total games learned from
        if hasattr(q_agent, "total_games_played"):
            total_games = q_agent.total_games_played
            q_table_size = len(q_agent.q_table) if hasattr(q_agent, "q_table") else 0
            st.info(f"**Total games learned from: {total_games} | Q-table size: {q_table_size} states | Epsilon: {q_agent.epsilon:.3f}**")
        
        # Show recent game results
        if hasattr(q_agent, "get_recent_games"):
            num_recent = st.slider("Show last N games", 1, 100, 25, key="num_recent_games")
            recent_games = q_agent.get_recent_games(num_recent)
            if recent_games:
                st.write(f"**Last {len(recent_games)} Games Results:**")
                wins_x = sum(1 for g in recent_games if g.get("winner") == 1)
                wins_o = sum(1 for g in recent_games if g.get("winner") == -1)
                draws = sum(1 for g in recent_games if g.get("winner") is None)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("X Wins", wins_x, f"{wins_x/len(recent_games)*100:.1f}%")
                with col2:
                    st.metric("O Wins", wins_o, f"{wins_o/len(recent_games)*100:.1f}%")
                with col3:
                    st.metric("Draws", draws, f"{draws/len(recent_games)*100:.1f}%")

    # Render board
    st.write("---")
    current_player = st.session_state.env.get_current_player()
    is_current_human = (current_player == 1 and is_human_x) or (
        current_player == -1 and is_human_o
    )
    
    board_disabled = (
        st.session_state.game_over
        or (not is_human_x and not is_human_o)
        or (is_current_human is False and (is_human_x or is_human_o))
    )
    
    selected_move = components.render_board(
        st.session_state.game_state,
        disabled=board_disabled,
    )

    # Handle human move from board click
    if selected_move is not None and not st.session_state.game_over:
        if is_current_human:
            try:
                state_before = st.session_state.game_state
                st.session_state.game_state, reward, done = st.session_state.env.step(selected_move)
                if done:
                    st.session_state.game_over = True
                    # Finalize game for Q-learning agents when human game ends
                    winner = st.session_state.env.get_winner()
                    if hasattr(agent_x, "finalize_game"):
                        agent_x.finalize_game(winner)
                        if hasattr(agent_x, "save"):
                            try:
                                agent_x.save("models/q_table.json")
                            except Exception:
                                pass
                    if hasattr(agent_o, "finalize_game"):
                        agent_o.finalize_game(winner)
                        if hasattr(agent_o, "save"):
                            try:
                                agent_o.save("models/q_table.json")
                            except Exception:
                                pass
                st.rerun()
            except ValueError:
                st.error("Invalid move. Please try again.")

    # Game status
    current_player = st.session_state.env.get_current_player()
    winner = st.session_state.env.get_winner()
    is_draw = (
        st.session_state.env.is_terminal()
        and winner is None
    )

    components.render_game_status(
        st.session_state.game_state,
        current_player,
        winner,
        is_draw,
    )

    # Game controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Reset Game"):
            st.session_state.env.reset()
            st.session_state.game_state = st.session_state.env.state
            st.session_state.game_over = False
            # Don't clear Q-learning history - only reset the game board
            st.rerun()

    with col2:
        if st.button("Make AI Move") and not st.session_state.game_over:
            _make_ai_move(agent_x, agent_o, is_human_x, is_human_o)
            st.rerun()


    # Q-Learning game state visualization (show even if human is playing)
    if is_qlearning_selected:
        q_agent = agent_x if player_x_type == "Q-Learning" else agent_o
        if hasattr(q_agent, "get_last_game"):
            last_game = q_agent.get_last_game()
            if last_game:
                with st.expander("See last GameState", expanded=False):
                    _render_game_state_progression(last_game, q_agent)
            else:
                # Show message if no game history yet
                st.info("No game history yet. Play a game to see Q-value updates!")
    


def _get_agent(agent_type: str, player_key: str):
    """Get agent instance based on type.

    Args:
        agent_type: Agent type name.
        player_key: Unique key for agent instance.

    Returns:
        Agent instance.
    """
    from app.agents.minimax_agent import MinimaxAgent
    from app.agents.mcts_agent import MCTSAgent
    from app.agents.q_learning_agent import QLearningAgent
    from app.agents.random_agent import RandomAgent
    from app.agents.rule_based_agent import RuleBasedAgent

    cache_key = f"agent_{player_key}_{agent_type}"

    if cache_key not in st.session_state:
        if agent_type == "Random":
            st.session_state[cache_key] = RandomAgent()
        elif agent_type == "Rule-Based":
            st.session_state[cache_key] = RuleBasedAgent()
        elif agent_type == "Minimax":
            st.session_state[cache_key] = MinimaxAgent()
        elif agent_type == "MCTS":
            st.session_state[cache_key] = MCTSAgent(num_simulations=500)
        elif agent_type == "Q-Learning":
            agent = QLearningAgent()
            # Try to load saved model
            try:
                agent.load("models/q_table.json")
            except Exception:
                pass
            st.session_state[cache_key] = agent

    return st.session_state[cache_key]


def _make_ai_move(agent_x, agent_o, is_human_x: bool, is_human_o: bool):
    """Make a move using the appropriate AI agent."""
    if st.session_state.game_over:
        return

    current_player = st.session_state.env.get_current_player()
    state = st.session_state.game_state

    if current_player == 1 and not is_human_x:
        # X's turn and X is AI
        move = agent_x.select_action(state, 1)
        next_state, reward, done = st.session_state.env.step(move)
        # Update Q-learning agent (if applicable)
        if hasattr(agent_x, "update"):
            agent_x.update(state, move, reward, next_state, 1, done)
        st.session_state.game_state = next_state
        if done:
            st.session_state.game_over = True
            if hasattr(agent_x, "finalize_game"):
                agent_x.finalize_game(st.session_state.env.get_winner())
                # Save Q-table
                if hasattr(agent_x, "save"):
                    try:
                        agent_x.save("models/q_table.json")
                    except Exception:
                        pass
    elif current_player == -1 and not is_human_o:
        # O's turn and O is AI
        move = agent_o.select_action(state, -1)
        next_state, reward, done = st.session_state.env.step(move)
        # Update Q-learning agent (if applicable)
        if hasattr(agent_o, "update"):
            agent_o.update(state, move, reward, next_state, -1, done)
        st.session_state.game_state = next_state
        if done:
            st.session_state.game_over = True
            if hasattr(agent_o, "finalize_game"):
                agent_o.finalize_game(st.session_state.env.get_winner())
                # Save Q-table
                if hasattr(agent_o, "save"):
                    try:
                        agent_o.save("models/q_table.json")
                    except Exception:
                        pass


def render_learn_page():
    """Render the algorithms explanation page."""
    st.header("Learn the Algorithms")

    algorithm = st.selectbox(
        "Select Algorithm",
        ["Random", "Rule-Based", "Minimax", "MCTS", "Q-Learning"],
    )

    st.write("---")

    if algorithm == "Random":
        _render_random_explanation()
    elif algorithm == "Rule-Based":
        _render_rule_based_explanation()
    elif algorithm == "Minimax":
        _render_minimax_explanation()
    elif algorithm == "MCTS":
        _render_mcts_explanation()
    elif algorithm == "Q-Learning":
        _render_qlearning_explanation()


def _render_random_explanation():
    """Render explanation for random agent."""
    st.subheader("Random Agent")
    st.write(
        "The random agent selects any legal move uniformly at random. "
        "This serves as a simple baseline for comparison. It has no strategy "
        "and will lose to any agent that uses even basic heuristics."
    )


def _render_rule_based_explanation():
    """Render explanation for rule-based agent."""
    st.subheader("Rule-Based Agent")
    st.write(
        "The rule-based agent uses simple heuristics to make decisions. "
        "It follows a priority list:"
    )
    st.write("1. **Take winning move**: If the agent can win in one move, it takes it.")
    st.write(
        "2. **Block opponent**: If the opponent can win next turn, block that move."
    )
    st.write(
        "3. **Prefer strategic positions**: Choose center first, then corners, then edges."
    )
    st.write(
        "This agent performs reasonably well against random play but will lose "
        "to perfect play algorithms like minimax."
    )


def _render_minimax_explanation():
    """Render explanation for minimax agent."""
    st.subheader("Minimax Agent")
    st.write(
        "Minimax is a decision-making algorithm for turn-based games. It assumes "
        "both players play optimally and evaluates all possible game outcomes."
    )
    st.write("**How it works:**")
    st.write(
        "1. Build a game tree of all possible moves and outcomes.\n"
        "2. At each level, alternate between maximizing (our turn) and minimizing (opponent's turn).\n"
        "3. Choose the move that leads to the best outcome assuming optimal play from both sides."
    )
    st.write("**Minimax recurrence:**")
    st.latex(
        r"""
        \text{minimax}(s, \text{player}) = \begin{cases}
        \text{utility}(s) & \text{if terminal} \\
        \max_{a} \text{minimax}(\text{result}(s, a), -\text{player}) & \text{if player = max} \\
        \min_{a} \text{minimax}(\text{result}(s, a), -\text{player}) & \text{if player = min}
        \end{cases}
        """
    )
    st.write(
        "**Alpha-beta pruning** improves efficiency by skipping branches that "
        "cannot improve the best move found so far. This doesn't change the result "
        "but makes the search much faster."
    )
    st.write(
        "For Tic-Tac-Toe, minimax with full depth search plays perfectly and will "
        "never lose (it will win or draw)."
    )


def _render_mcts_explanation():
    """Render explanation for MCTS agent."""
    st.subheader("Monte Carlo Tree Search (MCTS)")
    st.write(
        "MCTS builds a search tree by repeatedly simulating games. Unlike minimax, "
        "it doesn't explore the entire tree but focuses on promising branches."
    )
    st.write("**MCTS has four phases:**")
    st.write(
        "1. **Selection**: Traverse the tree using UCB1 formula to balance exploration and exploitation.\n"
        "2. **Expansion**: Add a new child node to the tree.\n"
        "3. **Simulation**: Play a random game from the new node to a terminal state.\n"
        "4. **Backpropagation**: Update node statistics (visits, wins) up the tree."
    )
    st.write("**UCB1 formula:**")
    st.latex(
        r"""
        \text{UCB1} = \frac{w_i}{n_i} + c \sqrt{\frac{\ln N}{n_i}}
        """
    )
    st.write(
        "where $w_i$ is wins, $n_i$ is visits, $N$ is parent visits, and $c$ is exploration constant."
    )
    st.write(
        "MCTS is particularly powerful in games with large state spaces where "
        "exhaustive search is impractical. With enough simulations, it approaches "
        "optimal play."
    )
    
    st.write("---")
    st.subheader("Interactive MCTS Tree Visualization")
    st.write("**Note:** MCTS doesn't train like Q-learning. It builds a fresh search tree for each move. This visualization shows how MCTS explores the game state and calculates UCB1 values at each node.")
    
    from app.agents.mcts_agent import MCTSAgent
    from app.game.environment import GameEnvironment
    
    # Create a fresh MCTS agent and environment for visualization
    mcts_agent = MCTSAgent(num_simulations=1000)
    env = GameEnvironment()
    initial_state = env.reset()
    
    mcts_num_sim = st.number_input(
        "Number of MCTS simulations per move",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="mcts_learn_simulations"
    )
    
    if st.button("Build & Visualize MCTS Tree", key="build_mcts_tree_learn"):
        with st.spinner(f"Building tree with {mcts_num_sim} simulations..."):
            _simulate_mcts_for_visualization(mcts_agent, mcts_num_sim)
        st.session_state.mcts_tree_built_learn = True
        st.rerun()
    
    if st.session_state.get("mcts_tree_built_learn", False):
        with st.expander("View MCTS Tree", expanded=True):
            _render_mcts_tree_visualization(mcts_agent, initial_state, mcts_num_sim)


def _render_qlearning_explanation():
    """Render explanation for Q-learning agent."""
    st.subheader("Q-Learning Agent")
    st.write(
        "Q-learning is a reinforcement learning algorithm that learns to make "
        "good decisions through trial and error. It maintains a Q-table that maps "
        "(state, action) pairs to expected future rewards."
    )
    st.write("**Q-learning update rule:**")
    st.latex(
        r"""
        Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
        """
    )
    st.write("where:")
    st.write("- $\\alpha$ (alpha) is the learning rate - controls how much we update")
    st.write(
        "- $\\gamma$ (gamma) is the discount factor - how much we value future rewards"
    )
    st.write("- $r$ is the immediate reward")
    st.write("- $s'$ is the next state after taking action $a$")
    st.write(
        "**Epsilon-greedy exploration**: The agent explores random moves with "
        "probability $\\epsilon$ and exploits the best known move otherwise. "
        "Epsilon typically decays over time to shift from exploration to exploitation."
    )
    st.write(
        "**Training**: The agent learns by playing many games against an opponent "
        "(often a random agent initially). Over time, it learns which moves lead "
        "to wins and which lead to losses."
    )
    st.write(
        "For Tic-Tac-Toe, the state space is small enough that tabular Q-learning "
        "can learn near-optimal play with sufficient training."
    )


def render_experiments_page():
    """Render the experiments/self-play page."""
    st.header("Experiments / Self-Play")

    st.write("Compare agents or train Q-learning through self-play.")

    experiment_type = st.radio(
        "Experiment Type",
        ["Agent Comparison", "Train Q-Learning"],
    )

    if experiment_type == "Agent Comparison":
        _render_agent_comparison()
    else:
        _render_qlearning_training()


def _render_agent_comparison():
    """Render agent comparison interface."""
    st.subheader("Compare Two Agents")

    col1, col2 = st.columns(2)
    with col1:
        agent_x_type = components.render_agent_selector("comp_x", "Agent X")
    with col2:
        agent_o_type = components.render_agent_selector("comp_o", "Agent O")

    num_games = st.slider("Number of Games", 10, 1000, 100, 10)

    if st.button("Run Comparison"):
        from app.agents.minimax_agent import MinimaxAgent
        from app.agents.mcts_agent import MCTSAgent
        from app.agents.q_learning_agent import QLearningAgent
        from app.agents.random_agent import RandomAgent
        from app.agents.rule_based_agent import RuleBasedAgent
        from app.evaluation.metrics import evaluate_agents

        agent_x = _create_agent(agent_x_type)
        agent_o = _create_agent(agent_o_type)

        with st.spinner(f"Running {num_games} games..."):
            metrics, _ = evaluate_agents(agent_x, agent_o, num_games)

        st.write("---")
        st.subheader("Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("X Wins", metrics.wins_x, f"{metrics.get_win_rate_x()*100:.1f}%")
        with col2:
            st.metric("O Wins", metrics.wins_o, f"{metrics.get_win_rate_o()*100:.1f}%")
        with col3:
            st.metric("Draws", metrics.draws, f"{metrics.get_draw_rate()*100:.1f}%")


def _render_qlearning_training():
    """Render Q-learning training interface."""
    st.subheader("Train Q-Learning Agent")

    col1, col2 = st.columns(2)
    with col1:
        learning_rate = st.slider("Learning Rate (α)", 0.01, 1.0, 0.1, 0.01)
        discount_factor = st.slider("Discount Factor (γ)", 0.1, 1.0, 0.9, 0.05)
    with col2:
        initial_epsilon = st.slider("Initial Epsilon (ε)", 0.1, 1.0, 0.5, 0.05)
        epsilon_decay = st.slider("Epsilon Decay", 0.9, 1.0, 0.995, 0.001)

    num_episodes = st.slider("Training Episodes", 100, 5000, 1000, 100)

    if st.button("Start Training"):
        from app.agents.q_learning_agent import QLearningAgent
        from app.agents.random_agent import RandomAgent
        from app.evaluation.self_play import train_q_learning

        agent = QLearningAgent(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
        )
        opponent = RandomAgent()

        progress_bar = st.progress(0)
        status_text = st.empty()

        training_history = []
        for i in range(num_episodes):
            episode_history = train_q_learning(agent, opponent, 1, verbose=False)
            training_history.extend(episode_history)
            progress = (i + 1) / num_episodes
            progress_bar.progress(progress)
            status_text.text(f"Episode {i+1}/{num_episodes}")

        # Save model
        import os

        os.makedirs("models", exist_ok=True)
        agent.save("models/q_table.json")

        st.success("Training complete! Model saved to models/q_table.json")

        # Plot training curve
        if training_history:
            import matplotlib.pyplot as plt

            episodes = [h["episode"] for h in training_history]
            rewards = [h["reward"] for h in training_history]
            epsilons = [h["epsilon"] for h in training_history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(episodes, rewards, alpha=0.3, label="Episode Reward")
            # Moving average
            window = 50
            if len(rewards) >= window:
                moving_avg = [
                    sum(rewards[max(0, i - window) : i + 1])
                    / min(window + 1, i + 1)
                    for i in range(len(rewards))
                ]
                ax1.plot(episodes, moving_avg, label=f"Moving Avg ({window})", linewidth=2)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Training Rewards")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(episodes, epsilons)
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Epsilon")
            ax2.set_title("Exploration Rate")
            ax2.grid(True, alpha=0.3)

            st.pyplot(fig)


def _create_agent(agent_type: str):
    """Create agent instance.

    Args:
        agent_type: Agent type name.

    Returns:
        Agent instance.
    """
    from app.agents.minimax_agent import MinimaxAgent
    from app.agents.mcts_agent import MCTSAgent
    from app.agents.q_learning_agent import QLearningAgent
    from app.agents.random_agent import RandomAgent
    from app.agents.rule_based_agent import RuleBasedAgent

    if agent_type == "Random":
        return RandomAgent()
    elif agent_type == "Rule-Based":
        return RuleBasedAgent()
    elif agent_type == "Minimax":
        return MinimaxAgent()
    elif agent_type == "MCTS":
        return MCTSAgent(num_simulations=500)
    elif agent_type == "Q-Learning":
        agent = QLearningAgent()
        try:
            agent.load("models/q_table.json")
        except Exception:
            pass
        return agent
    return RandomAgent()




def _simulate_games(agent_x, agent_o, player_x_type: str, player_o_type: str, num_games: int):
    """Simulate games between two agents.
    
    Each game randomly picks who goes first (X or O) so Q-learning
    learns from both perspectives.
    """
    from app.game.environment import GameEnvironment
    from app.evaluation.metrics import GameMetrics

    # Enable tracking for Q-learning agents (for UI display)
    if player_x_type == "Q-Learning" and hasattr(agent_x, "enable_history_tracking"):
        if not agent_x.track_history:
            agent_x.enable_history_tracking()
    if player_o_type == "Q-Learning" and hasattr(agent_o, "enable_history_tracking"):
        if not agent_o.track_history:
            agent_o.enable_history_tracking()

    metrics = GameMetrics()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Track who went first in each game
    first_player_stats = {"X": 0, "O": 0}
    
    for game_num in range(num_games):
        env = GameEnvironment()
        
        # Random first player (50/50 chance) - this helps Q-learning learn from both perspectives
        first_player = random.choice([1, -1])
        first_player_stats["X" if first_player == 1 else "O"] += 1
        state = env.reset()
        # Set who goes first (reset defaults to X, so only change if O should go first)
        if first_player == -1:
            env.current_player = -1

        # Play game until it ends
        while not env.is_terminal():
            current_player = env.get_current_player()
            if current_player == 1:
                # X's turn
                move = agent_x.select_action(state, 1)
                next_state, reward, done = env.step(move)
                # Update Q-learning agent (if applicable)
                if hasattr(agent_x, "update"):
                    agent_x.update(state, move, reward, next_state, 1, done)
                state = next_state
            else:
                # O's turn
                move = agent_o.select_action(state, -1)
                next_state, reward, done = env.step(move)
                # Update Q-learning agent (if applicable)
                if hasattr(agent_o, "update"):
                    agent_o.update(state, move, reward, next_state, -1, done)
                state = next_state

            if done:
                break

        # Record result and finalize Q-learning updates
        winner = env.get_winner()
        metrics.record_result(winner)

        if hasattr(agent_x, "finalize_game"):
            agent_x.finalize_game(winner)
        if hasattr(agent_o, "finalize_game"):
            agent_o.finalize_game(winner)

        # Update progress bar
        if (game_num + 1) % max(1, num_games // 20) == 0 or (game_num + 1) == num_games:
            progress = (game_num + 1) / num_games
            progress_bar.progress(progress)
            status_text.text(f"Completed {game_num + 1}/{num_games} games")

    # Save Q-learning models if applicable
    if hasattr(agent_x, "save") and player_x_type == "Q-Learning":
        try:
            agent_x.save("models/q_table.json")
        except Exception:
            pass
    if hasattr(agent_o, "save") and player_o_type == "Q-Learning":
        try:
            agent_o.save("models/q_table.json")
        except Exception:
            pass

    # Store results in session state so they persist
    st.session_state.simulation_results = {
        "wins_x": metrics.wins_x,
        "wins_o": metrics.wins_o,
        "draws": metrics.draws,
        "win_rate_x": metrics.get_win_rate_x(),
        "win_rate_o": metrics.get_win_rate_o(),
        "draw_rate": metrics.get_draw_rate(),
        "total_games": num_games,
        "first_player_x": first_player_stats["X"],
        "first_player_o": first_player_stats["O"],
        "player_x_type": player_x_type,
        "player_o_type": player_o_type,
    }
    
    # Show Q-learning specific info in results
    if player_x_type == "Q-Learning" and hasattr(agent_x, "total_games_played"):
        st.session_state.simulation_results["qlearning_x_games"] = agent_x.total_games_played
    if player_o_type == "Q-Learning" and hasattr(agent_o, "total_games_played"):
        st.session_state.simulation_results["qlearning_o_games"] = agent_o.total_games_played
    
    st.success(f"Simulation complete! {num_games} games played.")


def _render_game_state_progression(game_steps: list, q_agent):
    """Render game state progression with Q-table updates and backward propagation.

    Args:
        game_steps: List of step dictionaries from game history.
        q_agent: Q-learning agent instance.
    """
    import pandas as pd
    
    st.write(f"**Game Progression ({len(game_steps)} moves)**")
    
    if game_steps:
        result = game_steps[0].get("game_result")
        result_text = "Draw" if result is None else ("X Wins" if result == 1 else "O Wins")
        st.write(f"**Result:** {result_text}")
        
        # Show backward propagation summary table
        if any(step.get("is_backward_prop", False) for step in game_steps):
            st.write("---")
            st.subheader("Backward Propagation Summary")
            st.write("""
            **What is Backward Propagation?**
            
            At the end of the game, we know the final result (win/loss/draw). 
            We work backwards from the last move to the first, updating Q-values 
            using the formula:
            
            **target_q = 0 + γ × future_value**
            
            where:
            - **γ (gamma)** = discount factor = 0.9 (how much we value future rewards)
            - **future_value** = Q-value from the next move (propagated backward)
            """)
            
            # Create backward propagation table
            bp_data = []
            # Get agent's moves in reverse order (backward propagation order)
            agent_moves = [s for s in reversed(game_steps) if s.get("is_backward_prop", False)]
            
            for idx, step in enumerate(agent_moves):
                move_num = len(agent_moves) - idx
                future_val = step.get("future_value", 0.0)
                target_q = step.get("target_q", 0.0)
                q_before = step.get("q_value_before", 0.0)
                q_after = step.get("q_value_after", 0.0)
                learning_rate = q_agent.learning_rate if hasattr(q_agent, "learning_rate") else 0.1
                discount_factor = q_agent.discount_factor if hasattr(q_agent, "discount_factor") else 0.9
                
                # Calculate formula components - get the future_value that was actually used
                future_value_used = step.get("future_value_used")
                if future_value_used is None:
                    # Reconstruct from target_q if not stored
                    future_value_used = target_q / discount_factor if discount_factor > 0 else 0.0
                
                if idx == 0:
                    # Last move - future_value_used is the final reward
                    result_val = step.get("game_result")
                    if result_val == 1:
                        final_reward = 1.0
                    elif result_val == -1:
                        final_reward = -1.0
                    else:
                        final_reward = 0.0
                    calculated_target = discount_factor * final_reward
                    formula_explanation = f"γ × Final reward = {discount_factor:.1f} × {final_reward:.1f} = {calculated_target:.4f}"
                else:
                    # Earlier moves - future_value_used is previous move's Q-value after update
                    prev_q_after = agent_moves[idx-1].get("q_value_after", 0.0)
                    calculated_target = discount_factor * prev_q_after
                    formula_explanation = f"γ × Q(next move) = {discount_factor:.1f} × {prev_q_after:.4f} = {calculated_target:.4f}"
                
                # Update future_val display to show the value that was used
                future_val = future_value_used
                
                bp_data.append({
                    "Move": f"Move {move_num}",
                    "Future Value": f"{future_val:.4f}",
                    "Target Q Formula": formula_explanation,
                    "Target Q": f"{target_q:.4f}",
                    "Q Before": f"{q_before:.4f}",
                    "Q After": f"{q_after:.4f}",
                    "Update": f"{q_before:.4f} + {learning_rate:.1f}×({target_q:.4f}-{q_before:.4f})"
                })
            
            if bp_data:
                st.dataframe(pd.DataFrame(bp_data), use_container_width=True)
                st.caption("**Note:** Moves are shown in backward propagation order (last move first)")
    
    # Show detailed view for each move
    st.write("---")
    st.subheader("Detailed Move-by-Move Analysis")
    
    for i, step in enumerate(game_steps):
        st.write("---")
        st.write(f"**Move {i + 1}**")
        
        # Show board state
        state = step.get("state")
        if state:
            st.write("**Board State:**")
            st.code(state.to_string())
        
        # Show player and action
        player_id = step.get("player_id", 0)
        player_name = "X" if player_id == 1 else "O"
        action = step.get("action")
        is_exploration = step.get("is_exploration", False)
        
        st.write(f"**Player:** {player_name}")
        st.write(f"**Action:** Move {action} (row {action//3}, col {action%3})")
        st.write(f"**Strategy:** {'Exploration' if is_exploration else 'Exploitation'}")
        
        # Show Q-values before update
        q_values_before = step.get("q_values_before", {})
        valid_moves = step.get("valid_moves", [])
        
        # Show Q-values for all valid moves (even if not in q_values_before yet)
        st.write("**Q-values before update:**")
        q_df_data = []
        for move in valid_moves:
            row, col = divmod(move, 3)
            q_val = q_values_before.get(move, 0.0)
            q_df_data.append({
                "Move": move,
                "Position": f"({row}, {col})",
                "Q-value": f"{q_val:.4f}",
                "Selected": "✓" if move == action else ""
            })
        if q_df_data:
            st.dataframe(pd.DataFrame(q_df_data), use_container_width=True)
        
        # Show Q-value update if available
        if "q_value_before" in step and "q_value_after" in step:
            q_before = step.get("q_value_before", 0.0)
            q_after = step.get("q_value_after", 0.0)
            reward = step.get("reward", 0.0)
            target_q = step.get("target_q", 0.0)
            future_value = step.get("future_value")
            is_backward_prop = step.get("is_backward_prop", False)
            learning_rate = q_agent.learning_rate if hasattr(q_agent, "learning_rate") else 0.1
            discount_factor = q_agent.discount_factor if hasattr(q_agent, "discount_factor") else 0.9
            
            st.write("---")
            st.write("### Q-Learning Update Details")
            
            # Show the three key values clearly
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Reward (r)", f"{reward:.4f}")
                st.caption("""
                **What is Reward?**
                - Immediate feedback from the environment
                - +1.0 if you win
                - -1.0 if you lose  
                - 0.0 if draw or game continues
                """)
            with col2:
                st.metric("Target Q", f"{target_q:.4f}")
                st.caption("""
                **What is Target Q?**
                - The "goal" Q-value we want to learn
                - What the Q-value should be ideally
                - Calculated from reward + future value
                """)
            with col3:
                st.metric("Q-value", f"{q_before:.4f} → {q_after:.4f}")
                st.caption("""
                **What is Q-value?**
                - Stored value in Q-table
                - What we actually learned so far
                - Gets updated toward target_q
                """)
            
            st.write("---")
            
            # Show formulas and calculations
            if is_backward_prop:
                st.write("**Backward Propagation Update:**")
                st.write("""
                **Step 1: Calculate Target Q**
                
                Formula: **target_q = γ × future_value**
                
                Where:
                - **γ (gamma)** = discount factor = how much we value future rewards
                - **future_value** = value from the next move (used for this calculation)
                - Immediate reward = 0 (no reward during game, only at end)
                """)
                # Get the future_value that was actually used for this calculation
                future_value_used = step.get("future_value_used", future_value)
                # If not stored, reconstruct it from target_q
                if future_value_used is None:
                    future_value_used = target_q / discount_factor
                
                # Calculate target_q correctly for display
                calculated_target_q = discount_factor * future_value_used
                st.write(f"""
                **Calculation for this move:**
                - Discount factor (γ) = {discount_factor:.1f}
                - Future value used = {future_value_used:.4f} (from next move)
                - **target_q = {discount_factor:.1f} × {future_value_used:.4f} = {calculated_target_q:.4f}**
                """)
                # Verify the calculation matches stored value
                if abs(calculated_target_q - target_q) > 0.0001:
                    st.warning(f"Calculation mismatch: computed {calculated_target_q:.4f} but stored {target_q:.4f}")
            else:
                st.write("**Q-value Update (Forward):**")
                st.write("""
                **Step 1: Calculate Target Q**
                
                Formula: **target_q = reward + γ × max Q(next_state, action)**
                
                Where:
                - **reward (r)** = immediate reward from this move
                - **γ (gamma)** = discount factor = how much we value future rewards
                - **max Q(next_state)** = best Q-value in the next state
                """)
                st.write(f"""
                **Calculation for this move:**
                - Immediate reward (r) = {reward:.1f}
                - Discount factor (γ) = {discount_factor:.1f}
                - Max Q(next_state) = (calculated from next state)
                - **target_q = {reward:.1f} + {discount_factor:.1f} × max Q(s') = {target_q:.4f}**
                """)
            
            st.write("---")
            st.write("""
            **Step 2: Update Q-value**
            
            Formula: **Q(s,a) = Q(s,a) + α × (target_q - Q(s,a))**
            
            Where:
            - **Q(s,a)** = current Q-value stored in Q-table
            - **α (alpha)** = learning rate = how fast we learn (step size)
            - **target_q** = what we want to learn (from Step 1)
            """)
            
            # Show detailed calculation
            difference = target_q - q_before
            update_amount = learning_rate * difference
            st.write(f"""
            **Calculation for this move:**
            1. Current Q-value = {q_before:.4f}
            2. Target Q = {target_q:.4f} (from Step 1)
            3. Difference = target_q - Q(s,a) = {target_q:.4f} - {q_before:.4f} = {difference:.4f}
            4. Learning rate (α) = {learning_rate:.1f}
            5. Update amount = α × difference = {learning_rate:.1f} × {difference:.4f} = {update_amount:.4f}
            6. **New Q-value = {q_before:.4f} + {update_amount:.4f} = {q_after:.4f}**
            """)
            
            st.info(f"""
            **Summary:**
            - **Reward** = {reward:.4f} (what happened)
            - **Target Q** = {target_q:.4f} (what we want to learn)
            - **Q-value** = {q_before:.4f} → {q_after:.4f} (what we actually learned)
            
            The Q-value moved {abs(update_amount):.4f} closer to the target. 
            After many games, Q-value will gradually approach target_q.
            """)
        
        # Show Q-values after update
        q_values_after = step.get("q_values_after", {})
        if q_values_after and q_values_after != q_values_before:
            st.write("**Q-values after update:**")
            q_df_data = []
            for move, q_val in sorted(q_values_after.items()):
                row, col = divmod(move, 3)
                q_before_val = q_values_before.get(move, 0.0)
                change = q_val - q_before_val
                q_df_data.append({
                    "Move": move,
                    "Position": f"({row}, {col})",
                    "Q-value": f"{q_val:.4f}",
                    "Change": f"{change:+.4f}" if change != 0 else "0.0000"
                })
            if q_df_data:
                st.dataframe(pd.DataFrame(q_df_data), use_container_width=True)


def _render_mcts_tree_visualization(mcts_agent, initial_state, num_simulations: int = 1000):
    """Render MCTS tree visualization with UCB1 formulas and components.
    
    Args:
        mcts_agent: MCTS agent instance.
        initial_state: Starting game state.
        num_simulations: Number of simulations to run.
    """
    st.write("### MCTS Tree Visualization")
    st.write("This visualization shows how MCTS builds its search tree, including UCB1 calculations for each node.")
    
    # Build the tree
    with st.spinner(f"Building MCTS tree with {num_simulations} simulations..."):
        root = mcts_agent.build_tree_for_visualization(initial_state, num_simulations)
    
    st.success(f"Tree built! Root node has {len(root.children)} children.")
    
    # Get root action selector
    root_children = sorted(root.children, key=lambda c: c.visits, reverse=True)
    
    if not root_children:
        st.warning("No children found in root node. Tree may not have been built correctly.")
        return
    
    root_action_options = {f"Move {c.move} ({c.visits} visits)": c.move for c in root_children}
    root_action_labels = list(root_action_options.keys())
    
    # Default to center (move 4) if available, otherwise first child
    default_action = 4 if any(c.move == 4 for c in root_children) else root_children[0].move if root_children else 0
    
    # Find default index
    default_index = 0
    if default_action == 4:
        for i, (label, move) in enumerate(root_action_options.items()):
            if move == 4:
                default_index = i
                break
    else:
        for i, (label, move) in enumerate(root_action_options.items()):
            if move == default_action:
                default_index = i
                break
    
    selected_root_action = st.selectbox(
        "Select root action to explore (default: center)",
        options=root_action_labels,
        index=default_index,
        key="mcts_root_action"
    )
    
    # Get the move value from the selected label
    root_action = root_action_options.get(selected_root_action)
    if root_action is None:
        # Fallback to first child if selection fails
        root_action = root_children[0].move if root_children else 0
    
    # Get tree data
    tree_data = mcts_agent.get_tree_data(root_action=root_action, max_depth=2)
    
    if tree_data is None:
        st.warning("No tree data available. Please run simulations first.")
        return
    
    st.write("---")
    
    # Show root information
    root_info = tree_data["root"]
    st.write("**Root Node (Initial State):**")
    st.write(f"- Possible actions: {root_info['valid_moves_count']}")
    st.write(f"- Total visits: {root_info['visits']}")
    st.write(f"- Total wins: {root_info['wins']}")
    st.write(f"- Win rate: {root_info['win_rate']:.4f}")
    
    # Show all root children
    if root_info.get("root_children"):
        st.write("\n**All Root Actions:**")
        root_children_df = []
        for child_info in root_info["root_children"]:
            row, col = divmod(child_info["move"], 3)
            root_children_df.append({
                "Move": child_info["move"],
                "Position": f"({row}, {col})",
                "Visits": child_info["visits"],
                "Wins": child_info["wins"],
                "Win Rate": f"{child_info['wins']/child_info['visits']:.4f}" if child_info["visits"] > 0 else "0.0000"
            })
        if root_children_df:
            import pandas as pd
            st.dataframe(pd.DataFrame(root_children_df), use_container_width=True)
    
    st.write("---")
    st.write(f"**Tree starting from action {root_action}:**")
    
    # Render tree recursively with tables
    def render_node(node_data: Dict, depth: int = 0):
        """Render a node in the tree with table format."""
        import pandas as pd
        
        move = node_data["move"]
        position = node_data["position"]
        visits = node_data["visits"]
        wins = node_data["wins"]
        win_rate = node_data["win_rate"]
        ucb1 = node_data["ucb1"]
        exploitation = node_data["exploitation"]
        exploration_term = node_data["exploration_term"]
        parent_visits = node_data["parent_visits"]
        valid_moves = node_data["valid_moves_count"]
        
        # Create table for this node
        indent_prefix = "  " * depth
        st.write(f"{indent_prefix}**Move {move} at {position}**")
        
        # Build table data
        table_data = {
            "Metric": ["Exploitation (w/n)", "Exploration (c×√(ln(N)/n))", "N (Parent Visits)", "n (Node Visits)", "UCB1", "Wins", "Win Rate", "Possible Moves"],
            "Value": [
                f"{exploitation:.4f}" if exploitation is not None else "N/A",
                f"{exploration_term:.4f}" if exploration_term is not None and exploration_term != float("inf") else "∞" if exploration_term == float("inf") else "N/A",
                str(parent_visits),
                str(visits),
                f"{ucb1:.4f}" if ucb1 is not None and ucb1 != float("inf") else "∞" if ucb1 == float("inf") else "N/A",
                str(wins),
                f"{win_rate:.4f}",
                str(valid_moves)
            ]
        }
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.write("")
        
        # Render children
        if node_data.get("children"):
            for child in node_data["children"]:
                render_node(child, depth + 1)
    
    # Render the selected tree
    if tree_data.get("selected_tree"):
        render_node(tree_data["selected_tree"], depth=0)
    else:
        st.warning("No tree data for selected action.")


def _simulate_mcts_for_visualization(mcts_agent, num_simulations: int = 1000):
    """Simulate MCTS against random opponent to build tree.
    
    Args:
        mcts_agent: MCTS agent instance.
        num_simulations: Number of MCTS simulations per move (not games).
    """
    from app.game.environment import GameEnvironment
    
    env = GameEnvironment()
    state = env.reset()
    
    # Build tree from initial state
    mcts_agent.build_tree_for_visualization(state, num_simulations)
    
    return state

