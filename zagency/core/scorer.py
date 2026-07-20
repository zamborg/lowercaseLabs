"""
Scorer base classes for evaluating agent and environment behavior.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from zagency.core.agent import Agent
from zagency.core.environment import Environment


class Scorer(ABC):
    """
    Base class for all scorers. Scorers evaluate agents, environments, or traces
    to measure performance and behavior.
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self._scores: List[Dict[str, Any]] = []
    
    @abstractmethod
    def score(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Calculate and return a score. The return format should be a dictionary
        containing at least a 'score' key with the numeric score.
        """
        pass
    
    def reset(self):
        """Reset the scorer's internal state."""
        self._scores = []
    
    def get_scores(self) -> List[Dict[str, Any]]:
        """Get all recorded scores."""
        return self._scores.copy()
    
    def aggregate_scores(self, aggregation_fn: str = "mean") -> float:
        """
        Aggregate all recorded scores using the specified function.
        
        Args:
            aggregation_fn: One of "mean", "sum", "min", "max", "last"
        """
        if not self._scores:
            return 0.0
        
        score_values = [s.get("score", 0) for s in self._scores]
        
        if aggregation_fn == "mean":
            return sum(score_values) / len(score_values)
        elif aggregation_fn == "sum":
            return sum(score_values)
        elif aggregation_fn == "min":
            return min(score_values)
        elif aggregation_fn == "max":
            return max(score_values)
        elif aggregation_fn == "last":
            return score_values[-1]
        else:
            raise ValueError(f"Unknown aggregation function: {aggregation_fn}")


class AgentScorer(Scorer):
    """
    Scorer that evaluates agent behavior and internal state.
    AgentScorers have direct access to the agent's internal state and methods.
    """
    
    @abstractmethod
    def score(self, agent: Agent, step_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Score an agent based on its current state and optionally its last step result.
        
        Args:
            agent: The agent to score
            step_result: Optional result from the agent's last step
            
        Returns:
            Dictionary containing at least 'score' key with numeric value
        """
        pass


class EnvironmentScorer(Scorer):
    """
    Scorer that evaluates environment state and changes.
    EnvironmentScorers analyze the environment's state and API usage.
    """
    
    @abstractmethod
    def score(self, environment: Environment, agent: Optional[Agent] = None) -> Dict[str, Any]:
        """
        Score an environment based on its current state.
        
        Args:
            environment: The environment to score
            agent: Optional agent that triggered the scoring
            
        Returns:
            Dictionary containing at least 'score' key with numeric value
        """
        pass


class TraceScorer(Scorer):
    """
    Scorer that evaluates complete traces of agent-environment interactions.
    TraceScorers analyze the full history of interactions.
    """
    
    @abstractmethod
    def score(self, agent: Agent, environment: Environment, trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Score a complete trace of agent-environment interaction.
        
        Args:
            agent: The agent involved in the trace
            environment: The environment involved in the trace
            trace: Optional explicit trace data (if not provided, may use agent/env history)
            
        Returns:
            Dictionary containing at least 'score' key with numeric value
        """
        pass


# Example concrete scorer implementations

class ThinkingTokenScorer(AgentScorer):
    """
    Example AgentScorer that measures thinking token efficiency.
    """
    
    def score(self, agent: Agent, step_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate thinking token efficiency based on agent's internal state."""
        # Access agent's internal state for thinking tokens
        thinking_tokens = agent._internal_state.get("thinking_tokens", 0)
        total_tokens = agent._internal_state.get("total_tokens", 1)  # Avoid division by zero
        
        efficiency = 1.0 - (thinking_tokens / total_tokens) if total_tokens > 0 else 0.0
        
        result = {
            "score": efficiency,
            "thinking_tokens": thinking_tokens,
            "total_tokens": total_tokens,
            "scorer": self.name
        }
        
        self._scores.append(result)
        return result


class FileAccessScorer(EnvironmentScorer):
    """
    Example EnvironmentScorer that tracks file access patterns.
    """
    
    def __init__(self, name: str = None):
        super().__init__(name)
        self._file_access_count = {}
    
    def score(self, environment: Environment, agent: Optional[Agent] = None) -> Dict[str, Any]:
        """Calculate score based on file access patterns in the environment."""
        state = environment.get_state(agent)
        
        # Track file accesses (assuming environment tracks this)
        file_accesses = state.get("file_accesses", [])
        unique_files = set(file_accesses)
        
        # Penalize repeated access to same files
        redundancy_score = len(unique_files) / len(file_accesses) if file_accesses else 1.0
        
        result = {
            "score": redundancy_score,
            "total_accesses": len(file_accesses),
            "unique_files": len(unique_files),
            "blast_radius": len(unique_files),
            "scorer": self.name
        }
        
        self._scores.append(result)
        return result


class CompletionTimeScorer(TraceScorer):
    """
    Example TraceScorer that evaluates completion time and efficiency.
    """
    
    def score(self, agent: Agent, environment: Environment, trace: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Calculate score based on task completion time and steps."""
        # Use provided trace or construct from agent/environment history
        if trace is None:
            trace = agent.history
        
        num_steps = len(trace)
        is_complete = agent.is_finished
        
        # Score based on completion and efficiency
        if is_complete:
            # Fewer steps is better (normalized to 0-1 range)
            efficiency_score = 1.0 / (1.0 + num_steps / 10.0)
        else:
            efficiency_score = 0.0
        
        result = {
            "score": efficiency_score,
            "num_steps": num_steps,
            "completed": is_complete,
            "scorer": self.name
        }
        
        self._scores.append(result)
        return result