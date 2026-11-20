"""
Tests for evaluation framework.
"""

import pytest
from zagency.core.agent import Agent
from zagency.core.environment import SharedEnvironment
from zagency.core.lm import LM
from zagency.core.scorer import AgentScorer
from zagency.core.evaluation import Evaluation, EvaluationSuite
from zagency.handler.step_handler import StepHandler


class MockLM(LM):
    """Mock LM for testing."""
    def __init__(self):
        super().__init__(model="mock-model")
    
    def invoke(self, messages, tools=None):
        return {"response": type('obj', (object,), {"choices": [type('obj', (object,), {"message": type('obj', (object,), {"tool_calls": None})()})()]})()} 


class MockAgent(Agent):
    """Mock agent that completes after 2 steps."""
    def __init__(self, lm, environment):
        super().__init__(lm, environment)
        self.step_count = 0
    
    def step(self, environment):
        self.step_count += 1
        if self.step_count >= 2:
            self.is_finished = True
            return {"status": "completed"}
        return {"status": "working"}
    
    def ingest_state(self, environment):
        return environment.get_state()
    
    def synthesize_lm_input(self, **kwargs):
        return [{"role": "user", "content": "test"}]


class SimpleScorer(AgentScorer):
    """Simple scorer for testing."""
    def score(self, agent, step_result=None):
        score = 1.0 if agent.is_finished else 0.5
        result = {"score": score, "finished": agent.is_finished}
        self._scores.append(result)
        return result


def test_evaluation_single_datum():
    """Test evaluation on a single data point."""
    # Create evaluation
    evaluation = Evaluation(
        name="test_eval",
        runner=StepHandler(),
        scorers=[SimpleScorer()]
    )
    
    # Create agent
    lm = MockLM()
    env = SharedEnvironment()
    agent = MockAgent(lm, env)
    
    # Run on single datum
    datum = {"task": "test_task"}
    result = evaluation.run_single(
        agent=agent,
        environment_class=SharedEnvironment,
        datum=datum,
        max_steps=5
    )
    
    assert result["completed"] is True
    assert result["steps_taken"] == 2
    assert "initial_scores" in result
    assert "final_scores" in result
    assert "SimpleScorer" in result["final_scores"]


def test_evaluation_dataset():
    """Test evaluation on a dataset."""
    # Create evaluation
    evaluation = Evaluation(
        name="test_eval",
        runner=StepHandler(),
        scorers=[SimpleScorer()]
    )
    
    # Create agent
    lm = MockLM()
    env = SharedEnvironment()
    agent = MockAgent(lm, env)
    
    # Create dataset
    dataset = [
        {"task": "task1"},
        {"task": "task2"},
        {"task": "task3"}
    ]
    
    # Run evaluation
    results = evaluation.run(
        agent=agent,
        environment_class=SharedEnvironment,
        dataset=dataset,
        max_steps_per_datum=5
    )
    
    assert results["dataset_size"] == 3
    assert len(results["individual_results"]) == 3
    assert results["aggregate_scores"]["completion_rate"] == 1.0
    assert "SimpleScorer" in results["aggregate_scores"]["scorer_aggregates"]


def test_evaluation_suite():
    """Test evaluation suite with multiple evaluations."""
    # Create evaluations
    eval1 = Evaluation("eval1", scorers=[SimpleScorer()])
    eval2 = Evaluation("eval2", scorers=[SimpleScorer()])
    
    # Create suite
    suite = EvaluationSuite("test_suite")
    suite.add_evaluation(eval1)
    suite.add_evaluation(eval2)
    
    # Create agents
    lm = MockLM()
    env = SharedEnvironment()
    agent1 = MockAgent(lm, env)
    agent2 = MockAgent(lm, env)
    
    # Create datasets
    datasets = {
        "dataset1": [{"task": "task1"}, {"task": "task2"}],
        "dataset2": [{"task": "task3"}]
    }
    
    # Run suite
    results = suite.run(
        agents=[agent1, agent2],
        environment_class=SharedEnvironment,
        datasets=datasets,
        max_steps_per_datum=5
    )
    
    assert "evaluations" in results
    assert "eval1" in results["evaluations"]
    assert "eval2" in results["evaluations"]
    
    # Check structure
    eval1_results = results["evaluations"]["eval1"]
    assert "MockAgent" in eval1_results
    assert "dataset1" in eval1_results["MockAgent"]
    assert "dataset2" in eval1_results["MockAgent"]


def test_evaluation_aggregation():
    """Test custom aggregation function."""
    def custom_agg(results):
        # Sum all steps taken
        return sum(r["steps_taken"] for r in results)
    
    evaluation = Evaluation(
        name="test_eval",
        scorers=[SimpleScorer()],
        aggregation_fn=custom_agg
    )
    
    lm = MockLM()
    env = SharedEnvironment()
    agent = MockAgent(lm, env)
    
    dataset = [{"task": f"task{i}"} for i in range(3)]
    
    results = evaluation.run(
        agent=agent,
        environment_class=SharedEnvironment,
        dataset=dataset,
        max_steps_per_datum=5
    )
    
    # Each run takes 2 steps, 3 runs = 6 total
    assert results["aggregate_scores"]["custom_aggregate"] == 6