#!/usr/bin/env python3
"""
Runner for GrepAgent tasks - finds function definitions in repositories.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import weave

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.base import LiteLLM
from handler.step_handler import StepHandler
from grepAgent.grep_agent import GrepAgent
from grepAgent.grep_environment import GrepTaskEnvironment


class GrepTaskRunner:
    """Runs grep tasks for finding functions in repositories."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.results = []
    
    @weave.op()
    def run_single_task(self, 
                       function_name: str,
                       repo_url: str = None,
                       commit_hash: str = None,
                       local_path: str = None,
                       expected_filepath: str = None,
                       max_steps: int = 20) -> Dict[str, Any]:
        """
        Run a single grep task.
        
        Args:
            function_name: Name of function to find
            repo_url: Git repository URL
            commit_hash: Specific commit to search in
            local_path: Use local repository instead of cloning
            expected_filepath: Expected file (for evaluation)
            max_steps: Maximum agent steps
        """
        print(f"\n{'='*60}")
        print(f"Searching for function: {function_name}")
        if repo_url:
            print(f"Repository: {repo_url}")
        if commit_hash:
            print(f"Commit: {commit_hash}")
        if local_path:
            print(f"Local path: {local_path}")
        print(f"{'='*60}\n")
        
        try:
            # Create environment
            env = GrepTaskEnvironment(
                repo_url=repo_url,
                commit_hash=commit_hash,
                function_name=function_name,
                expected_filepath=expected_filepath,
                local_path=local_path
            )
            
            # Create agent
            lm = LiteLLM(model=self.model)
            agent = GrepAgent(lm, env)
            
            # Create handler and run
            handler = StepHandler(env)
            handler.add_agent(agent)
            
            # Run until task completed
            def stop_condition(state):
                return state.get("task_completed", False)
            
            handler.run(max_steps=max_steps, stop_condition=stop_condition)
            
            # Get final state and evaluation
            final_state = env.get_state()
            evaluation = env.evaluate()
            
            # Combine results
            result = {
                "function_name": function_name,
                "repo_url": repo_url,
                "commit_hash": commit_hash,
                "final_answer": final_state.get("final_answer"),
                "evaluation": evaluation,
                "total_commands": len(final_state["commands_executed"]),
                "commands": final_state["commands_executed"]
            }
            
            # Print summary
            self._print_result_summary(result)
            
            # Cleanup
            env.cleanup()
            
            return result
            
        except Exception as e:
            print(f"\nError running task: {str(e)}")
            return {
                "function_name": function_name,
                "error": str(e),
                "success": False
            }
    
    def run_dataset(self, dataset_path: str, max_steps: int = 20) -> List[Dict[str, Any]]:
        """
        Run grep tasks from a dataset file.
        
        Dataset should be JSON with entries containing:
        - function_name: Name of function to find
        - git_repo: Repository URL
        - git_hash: Commit hash
        - filepath: Expected file path (ground truth)
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        results = []
        
        for i, entry in enumerate(dataset):
            print(f"\n\nTask {i+1}/{len(dataset)}")
            
            result = self.run_single_task(
                function_name=entry["function_name"],
                repo_url=entry.get("git_repo"),
                commit_hash=entry.get("git_hash"),
                expected_filepath=entry.get("filepath"),
                max_steps=max_steps
            )
            
            results.append(result)
        
        # Print overall summary
        self._print_dataset_summary(results)
        
        return results
    
    def _print_result_summary(self, result: Dict[str, Any]):
        """Print summary of a single task result."""
        print("\n" + "-"*40)
        print("TASK RESULT")
        print("-"*40)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        answer = result.get("final_answer", {})
        if answer:
            print(f"Function: {result['function_name']}")
            print(f"Found in: {answer.get('filepath', 'Unknown')}")
            if answer.get('line_number'):
                print(f"Line: {answer['line_number']}")
        else:
            print("❌ No answer submitted")
        
        eval_result = result.get("evaluation", {})
        if eval_result:
            success = eval_result.get("success", False)
            print(f"\nEvaluation: {'✅ Correct' if success else '❌ Incorrect'}")
            if eval_result.get("expected"):
                print(f"Expected: {eval_result['expected']}")
            print(f"Score: {eval_result.get('score', 0):.2f}")
            
            metrics = eval_result.get("metrics", {})
            if metrics:
                print(f"\nMetrics:")
                print(f"  Total commands: {metrics.get('total_commands', 0)}")
                print(f"  Grep commands: {metrics.get('grep_commands', 0)}")
                print(f"  Duration: {metrics.get('duration_seconds', 0):.1f}s")
                print(f"  Efficiency: {metrics.get('efficiency', 0):.3f}")
    
    def _print_dataset_summary(self, results: List[Dict[str, Any]]):
        """Print summary of dataset results."""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        total = len(results)
        errors = sum(1 for r in results if "error" in r)
        completed = total - errors
        
        if completed > 0:
            correct = sum(1 for r in results 
                         if r.get("evaluation", {}).get("success", False))
            accuracy = correct / completed
            
            avg_commands = sum(r.get("total_commands", 0) for r in results 
                             if "error" not in r) / completed
            
            print(f"\nTotal tasks: {total}")
            print(f"Completed: {completed}")
            print(f"Errors: {errors}")
            print(f"Correct: {correct} ({accuracy:.1%})")
            print(f"Average commands per task: {avg_commands:.1f}")
        else:
            print(f"\nNo tasks completed successfully")
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to JSON file."""
        output_data = {
            "model": self.model,
            "total_tasks": len(results),
            "results": results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run GrepAgent to find functions in repositories")
    
    # Single task arguments
    parser.add_argument("--function", help="Function name to find")
    parser.add_argument("--repo", help="Git repository URL")
    parser.add_argument("--commit", help="Git commit hash")
    parser.add_argument("--local", help="Local repository path")
    parser.add_argument("--expected", help="Expected filepath (for evaluation)")
    
    # Dataset arguments
    parser.add_argument("--dataset", help="Path to dataset JSON file")
    
    # Common arguments
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum agent steps")
    parser.add_argument("--output", help="Output file for results")

    parser.add_argument("--project", help="Project name", default="wandb/zubin-dump")

    
    args = parser.parse_args()
    
    runner = GrepTaskRunner(model=args.model)

    weave.init(args.project)
    
    if args.dataset:
        # Run dataset
        results = runner.run_dataset(args.dataset, args.max_steps)
    elif args.function:
        # Run single task
        result = runner.run_single_task(
            function_name=args.function,
            repo_url=args.repo,
            commit_hash=args.commit,
            local_path=args.local,
            expected_filepath=args.expected,
            max_steps=args.max_steps
        )
        results = [result]
    else:
        print("Error: Specify either --function or --dataset")
        sys.exit(1)
    
    # Save results if requested
    if args.output:
        runner.save_results(results, args.output)


if __name__ == "__main__":
    main()