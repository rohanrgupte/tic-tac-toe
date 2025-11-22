"""Evaluation and self-play modules."""

from app.evaluation.metrics import GameMetrics, evaluate_agents
from app.evaluation.self_play import run_self_play, train_q_learning

__all__ = ["run_self_play", "train_q_learning", "evaluate_agents", "GameMetrics"]

