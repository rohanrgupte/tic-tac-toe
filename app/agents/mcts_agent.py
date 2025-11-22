"""Monte Carlo Tree Search agent."""

import math
import random
from typing import Dict, List, Optional, Tuple

from app.game.state import GameState


class MCTSNode:
    """Node in Monte Carlo Tree Search."""

    def __init__(self, state: GameState, parent: Optional["MCTSNode"] = None, move: Optional[int] = None):
        """Initialize MCTS node.

        Args:
            state: Game state at this node.
            parent: Parent node.
            move: Move that led to this state.
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.visits = 0
        self.wins = 0
        self.children: List["MCTSNode"] = []
        self.untried_moves = state.get_valid_moves()

    def is_fully_expanded(self) -> bool:
        """Check if all moves have been tried."""
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        """Check if state is terminal."""
        return self._check_winner() is not None or len(self.state.get_valid_moves()) == 0

    def ucb1_value(self, exploration: float = 1.414) -> float:
        """Calculate UCB1 value for node selection.

        UCB1 = wins/visits + c * sqrt(ln(parent_visits) / visits)

        Args:
            exploration: Exploration constant (default sqrt(2)).

        Returns:
            UCB1 value.
        """
        if self.visits == 0:
            return float("inf")

        exploitation = self.wins / self.visits
        parent_visits = self.parent.visits if self.parent else 1
        exploration_term = exploration * math.sqrt(math.log(parent_visits) / self.visits)

        return exploitation + exploration_term

    def best_child(self, exploration: float = 1.414) -> "MCTSNode":
        """Select best child using UCB1.

        Args:
            exploration: Exploration constant.

        Returns:
            Child node with highest UCB1 value.
        """
        return max(self.children, key=lambda c: c.ucb1_value(exploration))

    def expand(self) -> "MCTSNode":
        """Expand node by adding a new child.

        Returns:
            Newly created child node.
        """
        move = self.untried_moves.pop()
        next_state = self.state.apply_move(move, self._get_current_player())
        child = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def backpropagate(self, result: float):
        """Backpropagate simulation result up the tree.

        Args:
            result: Result from perspective of original player (1 for win, -1 for loss, 0 for draw).
        """
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)

    def _get_current_player(self) -> int:
        """Get current player based on move count.

        Returns:
            Player ID (1 for X, -1 for O).
        """
        # Count non-empty cells to determine current player
        non_empty = sum(1 for x in self.state.board.flatten() if x != 0)
        return 1 if non_empty % 2 == 0 else -1

    def _check_winner(self) -> Optional[int]:
        """Check if there is a winner.

        Returns:
            Player ID (1 or -1) if winner exists, None otherwise.
        """
        board = self.state.board

        # Check rows
        for row in range(3):
            if board[row, 0] != 0 and board[row, 0] == board[row, 1] == board[row, 2]:
                return board[row, 0]

        # Check columns
        for col in range(3):
            if board[0, col] != 0 and board[0, col] == board[1, col] == board[2, col]:
                return board[0, col]

        # Check diagonals
        if board[0, 0] != 0 and board[0, 0] == board[1, 1] == board[2, 2]:
            return board[0, 0]
        if board[0, 2] != 0 and board[0, 2] == board[1, 1] == board[2, 0]:
            return board[0, 2]

        return None


class MCTSAgent:
    """Agent using Monte Carlo Tree Search.

    MCTS builds a search tree by:
    1. Selection: Traverse tree using UCB1 to select promising nodes
    2. Expansion: Add new child node if not fully expanded
    3. Simulation: Play random game to terminal state
    4. Backpropagation: Update node statistics with result
    """

    def __init__(self, num_simulations: int = 1000):
        """Initialize MCTS agent.

        Args:
            num_simulations: Number of MCTS simulations per move.
        """
        self.name = "MCTS"
        self.num_simulations = num_simulations
        self.last_tree_root: Optional[MCTSNode] = None
        self.last_tree_root: Optional[MCTSNode] = None

    def select_action(self, state: GameState, player_id: int) -> int:
        """Select best move using MCTS.

        Args:
            state: Current game state.
            player_id: Player ID (1 for X, -1 for O).

        Returns:
            Best move index (0-8).
        """
        root = MCTSNode(state)

        for _ in range(self.num_simulations):
            # Selection: traverse to leaf
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion: add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation: play random game from current node's player perspective
            # Backpropagation will handle perspective flipping automatically
            result = self._simulate(node.state, node._get_current_player())

            # Backpropagation: update statistics (flips result at each level)
            node.backpropagate(result)

        # Select move with highest win rate (from root player's perspective)
        # Note: child nodes represent opponent's turn, so we want to minimize their win rate
        # which means maximizing -wins/visits (or minimizing wins/visits)
        if not root.children:
            valid_moves = state.get_valid_moves()
            return random.choice(valid_moves)

        # Select child with highest win rate from root's perspective
        # Since children are from opponent's perspective, we want the one with lowest wins/visits
        # (i.e., highest -wins/visits, which means best for root player)
        best_child = max(root.children, key=lambda c: -c.wins / c.visits if c.visits > 0 else float('-inf'))
        return best_child.move

    def _simulate(self, state: GameState, player_id: int) -> float:
        """Simulate random game from state to terminal.

        Args:
            state: Starting state.
            player_id: Current player.

        Returns:
            Result from player_id's perspective (1 for win, -1 for loss, 0 for draw).
        """
        current_state = state.copy()
        current_player = player_id

        while True:
            winner = self._check_winner(current_state)
            if winner is not None:
                return 1.0 if winner == player_id else -1.0

            valid_moves = current_state.get_valid_moves()
            if not valid_moves:
                return 0.0  # Draw

            move = random.choice(valid_moves)
            current_state = current_state.apply_move(move, current_player)
            current_player *= -1

    def _check_winner(self, state: GameState) -> Optional[int]:
        """Check if there is a winner.

        Returns:
            Player ID (1 or -1) if winner exists, None otherwise.
        """
        board = state.board

        # Check rows
        for row in range(3):
            if board[row, 0] != 0 and board[row, 0] == board[row, 1] == board[row, 2]:
                return board[row, 0]

        # Check columns
        for col in range(3):
            if board[0, col] != 0 and board[0, col] == board[1, col] == board[2, col]:
                return board[0, col]

        # Check diagonals
        if board[0, 0] != 0 and board[0, 0] == board[1, 1] == board[2, 2]:
            return board[0, 0]
        if board[0, 2] != 0 and board[0, 2] == board[1, 1] == board[2, 0]:
            return board[0, 2]

        return None

    def build_tree_for_visualization(self, state: GameState, num_simulations: int = None) -> MCTSNode:
        """Build MCTS tree for visualization purposes.
        
        Args:
            state: Starting game state.
            num_simulations: Number of simulations to run (uses self.num_simulations if None).
            
        Returns:
            Root node of the built tree.
        """
        if num_simulations is None:
            num_simulations = self.num_simulations
            
        root = MCTSNode(state)
        
        for _ in range(num_simulations):
            # Selection: traverse to leaf
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
            
            # Expansion: add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # Simulation: play random game from current node's player perspective
            # Backpropagation will handle perspective flipping automatically
            result = self._simulate(node.state, node._get_current_player())
            
            # Backpropagation: update statistics (flips result at each level)
            node.backpropagate(result)
        
        self.last_tree_root = root
        return root
    
    def get_tree_data(self, root_action: int = 4, max_depth: int = 2) -> Dict:
        """Get tree data for visualization.
        
        Args:
            root_action: The first action to show (default 4 = center).
            max_depth: Maximum depth to show (default 2).
            
        Returns:
            Dictionary with tree structure for visualization.
        """
        if self.last_tree_root is None:
            return None
        
        root = self.last_tree_root
        
        # Find the child node corresponding to root_action
        root_child = None
        for child in root.children:
            if child.move == root_action:
                root_child = child
                break
        
        if root_child is None:
            # If root_action not found, use first child
            if root.children:
                root_child = root.children[0]
            else:
                return None
        
        def node_to_dict(node: MCTSNode, depth: int, is_top_node: bool = False) -> Dict:
            """Convert node to dictionary for visualization."""
            if depth > max_depth:
                return None
            
            # Calculate UCB1 components
            exploration = 1.414
            if node.visits == 0:
                ucb1 = float("inf")
                exploitation = 0.0
                exploration_term = float("inf")
            else:
                exploitation = node.wins / node.visits
                parent_visits = node.parent.visits if node.parent else 1
                exploration_term = exploration * math.sqrt(math.log(parent_visits) / node.visits)
                ucb1 = exploitation + exploration_term
            
            # Get valid moves count
            valid_moves_count = len(node.state.get_valid_moves())
            
            # Get children: depth 0 shows all 8, depth 1 shows top 7 only for top 2 nodes
            children_data = []
            if depth < max_depth:
                sorted_children = sorted(node.children, key=lambda c: c.visits, reverse=True)
                if depth == 0:
                    # Show all 8 children from root action
                    max_children = 8
                    # Mark top 2 nodes for deeper expansion
                    for i, child in enumerate(sorted_children[:max_children]):
                        is_top = i < 2  # Top 2 nodes get expanded deeper
                        child_data = node_to_dict(child, depth + 1, is_top_node=is_top)
                        if child_data:
                            children_data.append(child_data)
                else:
                    # Depth 1: only show children if this is a top node
                    if is_top_node:
                        max_children = 7
                        for child in sorted_children[:max_children]:
                            child_data = node_to_dict(child, depth + 1, is_top_node=False)
                            if child_data:
                                children_data.append(child_data)
            
            row, col = divmod(node.move, 3) if node.move is not None else (None, None)
            
            return {
                "move": node.move,
                "position": f"({row}, {col})" if node.move is not None else "root",
                "visits": node.visits,
                "wins": node.wins,
                "win_rate": node.wins / node.visits if node.visits > 0 else 0.0,
                "ucb1": ucb1,
                "exploitation": exploitation,
                "exploration_term": exploration_term,
                "parent_visits": node.parent.visits if node.parent else 1,
                "valid_moves_count": valid_moves_count,
                "children": children_data,
                "depth": depth,
            }
        
        # Build root data
        root_valid_moves = len(root.state.get_valid_moves())
        root_data = {
            "move": None,
            "position": "root",
            "visits": root.visits,
            "wins": root.wins,
            "win_rate": root.wins / root.visits if root.visits > 0 else 0.0,
            "ucb1": None,
            "exploitation": None,
            "exploration_term": None,
            "parent_visits": 1,
            "valid_moves_count": root_valid_moves,
            "children": [],
            "depth": -1,
        }
        
        # Add all root children (showing their move info)
        root_children_info = []
        for child in sorted(root.children, key=lambda c: c.visits, reverse=True):
            row, col = divmod(child.move, 3)
            root_children_info.append({
                "move": child.move,
                "position": f"({row}, {col})",
                "visits": child.visits,
                "wins": child.wins,
            })
        root_data["root_children"] = root_children_info
        
        # Build tree from selected root action (mark as top node for expansion)
        selected_tree = node_to_dict(root_child, 0, is_top_node=True)
        
        return {
            "root": root_data,
            "selected_tree": selected_tree,
            "selected_action": root_action,
        }

