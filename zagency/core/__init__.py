"""
Core modules for the zagency framework.
"""

from zagency.core.base import tool
from zagency.core.agent import Agent
from zagency.core.environment import Environment, SharedEnvironment
from zagency.core.lm import LM
from zagency.core.scorer import (
    Scorer,
    AgentScorer,
    EnvironmentScorer,
    TraceScorer,
    ThinkingTokenScorer,
    FileAccessScorer,
    CompletionTimeScorer
)
from zagency.core.evaluation import Evaluation, EvaluationSuite

__all__ = [
    # Base
    "tool",
    
    # Core classes
    "Agent",
    "Environment",
    "SharedEnvironment", 
    "LM",
    
    # Scorer classes
    "Scorer",
    "AgentScorer",
    "EnvironmentScorer",
    "TraceScorer",
    "ThinkingTokenScorer",
    "FileAccessScorer",
    "CompletionTimeScorer",
    
    # Evaluation classes
    "Evaluation",
    "EvaluationSuite"
]