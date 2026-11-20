"""
Tests for scorer implementations.
"""

import pytest
from zagency.core.agent import Agent
from zagency.core.environment import SharedEnvironment
from zagency.core.lm import LM
from zagency.core.scorer import (
    Scorer, AgentScorer, EnvironmentScorer, TraceScorer,
    ThinkingTokenScorer, FileAccessScorer, CompletionTimeScorer
)


class MockLM(LM):
    """Mock LM for testing."""
    def __init__(self):
        super().__init__(model="mock-model")
    
    def invoke(self, messages, tools=None):
        return {"response": type('obj', (object,), {"choices": [type('obj', (object,), {"message": type('obj', (object,), {"tool_calls": None})()})()]})()} 


class MockAgent(Agent):
    """Mock agent for testing."""
    def step(self, environment):
        return {"status": "idle"}
    
    def ingest_state(self, environment):
        return environment.get_state()
    
    def synthesize_lm_input(self, **kwargs):
        return [{"role": "user", "content": "test"}]


def test_agent_scorer():
    """Test AgentScorer functionality."""
    lm = MockLM()
    env = SharedEnvironment()
    agent = MockAgent(lm, env)
    
    # Set some internal state
    agent._internal_state["thinking_tokens"] = 50
    agent._internal_state["total_tokens"] = 100
    
    scorer = ThinkingTokenScorer()
    result = scorer.score(agent)
    
    assert "score" in result
    assert result["score"] == 0.5  # 1.0 - (50/100)
    assert result["thinking_tokens"] == 50
    assert result["total_tokens"] == 100


def test_environment_scorer():
    """Test EnvironmentScorer functionality."""
    lm = MockLM()
    env = SharedEnvironment()
    agent = MockAgent(lm, env)
    
    # Set environment state
    env.update_state({"file_accesses": ["file1.py", "file2.py", "file1.py", "file3.py"]})
    
    scorer = FileAccessScorer()
    result = scorer.score(env, agent)
    
    assert "score" in result
    assert result["score"] == 0.75  # 3 unique files / 4 total accesses
    assert result["unique_files"] == 3
    assert result["blast_radius"] == 3


def test_trace_scorer():
    """Test TraceScorer functionality."""
    lm = MockLM()
    env = SharedEnvironment()
    agent = MockAgent(lm, env)
    
    # Simulate a completed trace
    agent.history = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "working"},
        {"role": "assistant", "content": "done"}
    ]
    agent.is_finished = True
    
    scorer = CompletionTimeScorer()
    result = scorer.score(agent, env)
    
    assert "score" in result
    assert result["completed"] is True
    assert result["num_steps"] == 3
    assert result["score"] > 0  # Should have a positive score for completion


def test_scorer_aggregation():
    """Test scorer aggregation methods."""
    scorer = ThinkingTokenScorer()
    
    # Add some scores
    scorer._scores = [
        {"score": 0.5},
        {"score": 0.7},
        {"score": 0.3}
    ]
    
    assert scorer.aggregate_scores("mean") == 0.5
    assert scorer.aggregate_scores("sum") == 1.5
    assert scorer.aggregate_scores("min") == 0.3
    assert scorer.aggregate_scores("max") == 0.7
    assert scorer.aggregate_scores("last") == 0.3


def test_scorer_reset():
    """Test scorer reset functionality."""
    scorer = ThinkingTokenScorer()
    scorer._scores = [{"score": 0.5}]
    
    assert len(scorer.get_scores()) == 1
    scorer.reset()
    assert len(scorer.get_scores()) == 0