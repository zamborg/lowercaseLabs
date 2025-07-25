"""
Evaluation framework for running agents on datasets with scorers.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Type
from zagency.core.agent import Agent
from zagency.core.environment import Environment
from zagency.core.scorer import Scorer, AgentScorer, EnvironmentScorer, TraceScorer
from zagency.handler.step_handler import StepHandler


class Evaluation:
    """
    Evaluation orchestrates running agents on datasets and collecting scores.
    
    Following the thesis notation:
    Υ(A_i, D_j, E_k) = Σ υ(A_i, E_k(d_u)) for d_u ∈ D_j
    
    Where:
    - A_i is an agent
    - D_j is a dataset
    - E_k is an environment type
    - υ is the row-wise evaluation function (runner + scorers)
    """
    
    def __init__(
        self,
        name: str,
        runner: Optional[StepHandler] = None,
        scorers: Optional[List[Scorer]] = None,
        aggregation_fn: Optional[Callable] = None
    ):
        """
        Initialize an evaluation.
        
        Args:
            name: Name of the evaluation
            runner: StepHandler to execute agents (defaults to basic StepHandler)
            scorers: List of scorers to apply
            aggregation_fn: Function to aggregate results across dataset (defaults to sum)
        """
        self.name = name
        self.runner = runner or StepHandler()
        self.scorers = scorers or []
        self.aggregation_fn = aggregation_fn or self._default_aggregation
        self.results: List[Dict[str, Any]] = []
    
    def add_scorer(self, scorer: Scorer):
        """Add a scorer to the evaluation."""
        self.scorers.append(scorer)
    
    def run(
        self,
        agent: Agent,
        environment_class: Type[Environment],
        dataset: List[Dict[str, Any]],
        max_steps_per_datum: int = 100,
        stop_condition: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the evaluation on a dataset.
        
        Args:
            agent: The agent to evaluate
            environment_class: Environment class to instantiate for each datum
            dataset: List of data points to seed environments
            max_steps_per_datum: Maximum steps per data point
            stop_condition: Optional custom stop condition
            
        Returns:
            Evaluation results including individual and aggregate scores
        """
        self.results = []
        
        for i, datum in enumerate(dataset):
            print(f"\n[Evaluation {self.name}] Running datum {i+1}/{len(dataset)}")
            
            # Run single datum evaluation
            datum_result = self.run_single(
                agent=agent,
                environment_class=environment_class,
                datum=datum,
                datum_id=i,
                max_steps=max_steps_per_datum,
                stop_condition=stop_condition
            )
            
            self.results.append(datum_result)
        
        # Aggregate results
        aggregate_scores = self._aggregate_results()
        
        return {
            "evaluation": self.name,
            "agent": agent.__class__.__name__,
            "dataset_size": len(dataset),
            "individual_results": self.results,
            "aggregate_scores": aggregate_scores
        }
    
    def run_single(
        self,
        agent: Agent,
        environment_class: Type[Environment],
        datum: Dict[str, Any],
        datum_id: Any = None,
        max_steps: int = 100,
        stop_condition: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on a single data point.
        This implements υ(A_i, E_k(d_u)) from the thesis.
        
        Args:
            agent: The agent to evaluate
            environment_class: Environment class to instantiate
            datum: Single data point to seed environment
            datum_id: Optional identifier for this datum
            max_steps: Maximum steps for this run
            stop_condition: Optional custom stop condition
            
        Returns:
            Results for this single datum including all scorer outputs
        """
        # Create fresh environment with datum as initial state
        environment = environment_class()
        if hasattr(environment, 'load_from_state'):
            environment.load_from_state(datum)
        else:
            environment.update_state(datum)
        
        # Reset agent for fresh run
        agent.environment = environment
        agent.is_finished = False
        agent.history = []
        agent._internal_state = {}
        # Reset any custom attributes
        if hasattr(agent, 'step_count'):
            agent.step_count = 0
        
        # Reset scorers
        for scorer in self.scorers:
            scorer.reset()
        
        # Setup runner with fresh environment
        self.runner.environment = environment
        self.runner.agents = [agent]
        self.runner.step_count = 0
        
        # Score initial state
        initial_scores = self._apply_scorers(agent, environment, step_result=None, is_initial=True)
        
        # Run agent to completion
        def wrapped_stop_condition(env_state):
            # Check agent completion
            if agent.is_finished:
                return True
            # Check custom condition
            if stop_condition:
                return stop_condition(env_state)
            return False
        
        # Execute the run
        self.runner.run(max_steps=max_steps, stop_condition=wrapped_stop_condition)
        
        # Score final state
        final_scores = self._apply_scorers(agent, environment, step_result=None, is_final=True)
        
        # Compile results
        return {
            "datum_id": datum_id,
            "datum": datum,
            "completed": agent.is_finished,
            "steps_taken": self.runner.step_count,
            "initial_scores": initial_scores,
            "final_scores": final_scores,
            "scorer_aggregates": self._get_scorer_aggregates()
        }
    
    def _apply_scorers(
        self,
        agent: Agent,
        environment: Environment,
        step_result: Optional[Dict[str, Any]] = None,
        is_initial: bool = False,
        is_final: bool = False
    ) -> Dict[str, Any]:
        """Apply all scorers and collect results."""
        scores = {}
        
        for scorer in self.scorers:
            try:
                if isinstance(scorer, AgentScorer):
                    score_result = scorer.score(agent, step_result)
                elif isinstance(scorer, EnvironmentScorer):
                    score_result = scorer.score(environment, agent)
                elif isinstance(scorer, TraceScorer):
                    # TraceScorers typically run at the end
                    if is_final:
                        score_result = scorer.score(agent, environment)
                    else:
                        continue
                else:
                    # Generic scorer - let it figure out what to do
                    score_result = scorer.score(
                        agent=agent,
                        environment=environment,
                        step_result=step_result
                    )
                
                scores[scorer.name] = score_result
                
            except Exception as e:
                scores[scorer.name] = {
                    "score": 0.0,
                    "error": str(e)
                }
        
        return scores
    
    def _get_scorer_aggregates(self) -> Dict[str, float]:
        """Get aggregate scores from each scorer."""
        aggregates = {}
        
        for scorer in self.scorers:
            aggregates[scorer.name] = scorer.aggregate_scores()
        
        return aggregates
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate results across all data points."""
        if not self.results:
            return {}
        
        # Completion rate
        completed_count = sum(1 for r in self.results if r["completed"])
        completion_rate = completed_count / len(self.results)
        
        # Average steps
        avg_steps = sum(r["steps_taken"] for r in self.results) / len(self.results)
        
        # Aggregate each scorer across all runs
        scorer_aggregates = {}
        for scorer in self.scorers:
            all_scores = []
            for result in self.results:
                scorer_result = result["final_scores"].get(scorer.name, {})
                if "score" in scorer_result:
                    all_scores.append(scorer_result["score"])
            
            if all_scores:
                scorer_aggregates[scorer.name] = {
                    "mean": sum(all_scores) / len(all_scores),
                    "min": min(all_scores),
                    "max": max(all_scores),
                    "sum": sum(all_scores)
                }
        
        return {
            "completion_rate": completion_rate,
            "average_steps": avg_steps,
            "scorer_aggregates": scorer_aggregates,
            "custom_aggregate": self.aggregation_fn(self.results) if self.aggregation_fn else None
        }
    
    @staticmethod
    def _default_aggregation(results: List[Dict[str, Any]]) -> float:
        """Default aggregation sums all final scores."""
        total = 0.0
        count = 0
        
        for result in results:
            for scorer_name, scorer_result in result["final_scores"].items():
                if "score" in scorer_result:
                    total += scorer_result["score"]
                    count += 1
        
        return total / count if count > 0 else 0.0


class EvaluationSuite:
    """
    A suite of evaluations to run multiple agents on multiple datasets.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.evaluations: List[Evaluation] = []
        self.results: Dict[str, Any] = {}
    
    def add_evaluation(self, evaluation: Evaluation):
        """Add an evaluation to the suite."""
        self.evaluations.append(evaluation)
    
    def run(
        self,
        agents: List[Agent],
        environment_class: Type[Environment],
        datasets: Dict[str, List[Dict[str, Any]]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run all evaluations on all agents and datasets.
        
        Args:
            agents: List of agents to evaluate
            environment_class: Environment class to use
            datasets: Dictionary mapping dataset names to data
            **kwargs: Additional arguments passed to evaluation runs
            
        Returns:
            Complete results for all evaluations
        """
        self.results = {
            "suite": self.name,
            "evaluations": {}
        }
        
        for evaluation in self.evaluations:
            eval_results = {}
            
            for agent in agents:
                agent_name = agent.__class__.__name__
                eval_results[agent_name] = {}
                
                for dataset_name, dataset in datasets.items():
                    print(f"\n[Suite {self.name}] Running {evaluation.name} - {agent_name} - {dataset_name}")
                    
                    result = evaluation.run(
                        agent=agent,
                        environment_class=environment_class,
                        dataset=dataset,
                        **kwargs
                    )
                    
                    eval_results[agent_name][dataset_name] = result
            
            self.results["evaluations"][evaluation.name] = eval_results
        
        return self.results
    
    def summarize(self) -> str:
        """Generate a summary of the evaluation suite results."""
        if not self.results:
            return "No results available"
        
        summary = [f"Evaluation Suite: {self.name}\n"]
        
        for eval_name, eval_results in self.results["evaluations"].items():
            summary.append(f"\nEvaluation: {eval_name}")
            
            for agent_name, agent_results in eval_results.items():
                summary.append(f"  Agent: {agent_name}")
                
                for dataset_name, dataset_results in agent_results.items():
                    agg = dataset_results["aggregate_scores"]
                    summary.append(f"    Dataset: {dataset_name}")
                    summary.append(f"      Completion Rate: {agg['completion_rate']:.2%}")
                    summary.append(f"      Average Steps: {agg['average_steps']:.1f}")
                    
                    for scorer_name, scorer_agg in agg["scorer_aggregates"].items():
                        summary.append(f"      {scorer_name}: {scorer_agg['mean']:.3f}")
        
        return "\n".join(summary)