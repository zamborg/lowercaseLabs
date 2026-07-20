"""
zagency - A framework for building and evaluating agents.
"""

from zagency.core import *
from zagency.handler import StepHandler, MultiAgentOrchestrator

__version__ = "0.1.0"

__all__ = [
    # Core exports
    "tool",
    "Agent", 
    "Environment",
    "SharedEnvironment",
    "LM",
    
    # Scorer exports
    "Scorer",
    "AgentScorer", 
    "EnvironmentScorer",
    "TraceScorer",
    
    # Evaluation exports
    "Evaluation",
    "EvaluationSuite",
    
    # Handler exports
    "StepHandler",
    "MultiAgentOrchestrator"
]