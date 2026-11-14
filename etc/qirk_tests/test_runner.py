#!/usr/bin/env python3
"""
Test runner for QirkTheCoder evaluation scenarios.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.base import LiteLLM
from agents.qirk_coder import QirkTheCoder
from handler.step_handler import StepHandler
from qirk_tests.environment_loader import EnvironmentLoader, TestEnvironment


class QirkTestRunner:
    """Runs evaluation scenarios for QirkTheCoder."""
    
    def __init__(self, model: str = "gpt-4-turbo", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.results = {}
    
    def run_scenario(self, scenario_path: str, max_steps: int = 20) -> Dict[str, Any]:
        """Run a single test scenario."""
        print(f"\n{'='*60}")
        print(f"Running scenario: {Path(scenario_path).stem}")
        print(f"{'='*60}\n")
        
        try:
            # Load the scenario
            env = EnvironmentLoader.load_scenario(scenario_path)
            config = env.scenario_config
            
            # Extract or infer tasks
            tasks = EnvironmentLoader.extract_tasks_from_config(config)
            
            # Set up initial tasks
            env.update_state({"pending_tasks": tasks})
            
            # Print scenario info
            metadata = config.get("metadata", {})
            print(f"Name: {metadata.get('name', 'Unknown')}")
            print(f"Description: {metadata.get('description', 'No description')}")
            print(f"Difficulty: {metadata.get('difficulty', 'Unknown')}")
            print(f"Tags: {', '.join(metadata.get('tags', []))}")
            print(f"\nTasks:")
            for i, task in enumerate(tasks, 1):
                print(f"  {i}. {task}")
            
            # Create agent
            lm = LiteLLM(model=self.model)
            qirk = QirkTheCoder(lm, env.project_root, env)
            
            # Create handler and run
            handler = StepHandler(env)
            handler.add_agent(qirk)
            
            # Run with stop condition
            def stop_condition(state):
                return not state.get("pending_tasks", [])
            
            start_time = datetime.now()
            handler.run(max_steps=max_steps, stop_condition=stop_condition)
            end_time = datetime.now()
            
            # Evaluate results
            evaluation = env.evaluate()
            
            # Add timing and step info
            evaluation["duration_seconds"] = (end_time - start_time).total_seconds()
            evaluation["total_steps"] = handler.step_count
            evaluation["max_steps_reached"] = handler.step_count >= max_steps
            
            # Print results
            self._print_evaluation_results(evaluation)
            
            return evaluation
            
        except Exception as e:
            print(f"\nError running scenario: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e),
                "score": 0
            }
    
    def run_all_scenarios(self, scenarios_dir: str, max_steps: int = 20) -> Dict[str, Dict[str, Any]]:
        """Run all scenarios in a directory."""
        scenarios_path = Path(scenarios_dir)
        results = {}
        
        # Find all scenario files
        scenario_files = sorted(scenarios_path.glob("*.yaml"))
        scenario_files = [f for f in scenario_files if f.name != "environment_schema.yaml"]
        
        print(f"Found {len(scenario_files)} scenarios to run")
        
        for scenario_file in scenario_files:
            scenario_name = scenario_file.stem
            result = self.run_scenario(str(scenario_file), max_steps)
            results[scenario_name] = result
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_evaluation_results(self, evaluation: Dict[str, Any]):
        """Print evaluation results in a nice format."""
        print("\n" + "-"*40)
        print("EVALUATION RESULTS")
        print("-"*40)
        
        # Overall result
        success = evaluation.get("success", False)
        print(f"Overall Result: {'✅ PASSED' if success else '❌ FAILED'}")
        print(f"Score: {evaluation.get('score', 0)} points")
        print(f"Duration: {evaluation.get('duration_seconds', 0):.1f} seconds")
        print(f"Steps taken: {evaluation.get('total_steps', 0)}")
        
        # Individual checks
        checks = evaluation.get("checks", {})
        if checks:
            print("\nIndividual Checks:")
            for check_name, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {check_name}")
        
        # Errors
        errors = evaluation.get("errors", [])
        if errors:
            print("\nErrors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
    
    def _print_summary(self, results: Dict[str, Dict[str, Any]]):
        """Print summary of all test results."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total = len(results)
        passed = sum(1 for r in results.values() if r.get("success", False))
        failed = total - passed
        
        print(f"\nTotal scenarios: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Calculate average metrics
        total_score = sum(r.get("score", 0) for r in results.values())
        total_duration = sum(r.get("duration_seconds", 0) for r in results.values())
        total_steps = sum(r.get("total_steps", 0) for r in results.values())
        
        print(f"\nAverage score: {total_score/total:.1f}")
        print(f"Total duration: {total_duration:.1f} seconds")
        print(f"Average steps per scenario: {total_steps/total:.1f}")
        
        # Show failed scenarios
        if failed > 0:
            print("\nFailed scenarios:")
            for name, result in results.items():
                if not result.get("success", False):
                    error = result.get("error", "Unknown error")
                    print(f"  - {name}: {error}")
    
    def save_results(self, results: Dict[str, Dict[str, Any]], output_path: str):
        """Save test results to a JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        output_data = {
            "test_run": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "total_scenarios": len(results),
                "passed": sum(1 for r in results.values() if r.get("success", False)),
                "failed": sum(1 for r in results.values() if not r.get("success", False))
            },
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run QirkTheCoder evaluation tests")
    parser.add_argument(
        "scenario",
        nargs="?",
        help="Path to specific scenario YAML file or directory of scenarios"
    )
    parser.add_argument(
        "--model",
        default="gpt-4-turbo",
        help="Model to use for testing (default: gpt-4-turbo)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum steps per scenario (default: 20)"
    )
    parser.add_argument(
        "--output",
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = QirkTestRunner(model=args.model, verbose=args.verbose)
    
    # Determine what to run
    if not args.scenario:
        # Run all scenarios in default directory
        scenario_path = Path(__file__).parent / "scenarios"
        results = runner.run_all_scenarios(str(scenario_path), args.max_steps)
    else:
        scenario_path = Path(args.scenario)
        if scenario_path.is_dir():
            # Run all scenarios in directory
            results = runner.run_all_scenarios(str(scenario_path), args.max_steps)
        else:
            # Run single scenario
            result = runner.run_scenario(str(scenario_path), args.max_steps)
            results = {scenario_path.stem: result}
    
    # Save results if requested
    if args.output:
        runner.save_results(results, args.output)
    
    # Exit with appropriate code
    all_passed = all(r.get("success", False) for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()