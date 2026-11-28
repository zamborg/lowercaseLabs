"""
Environment Loader for loading test scenarios from YAML files.
"""

import os
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from environments.coding_environment import CodingEnvironment


class TestEnvironment(CodingEnvironment):
    """Extended CodingEnvironment for testing with evaluation capabilities."""
    
    def __init__(self, project_root: str, scenario_config: Dict[str, Any]):
        super().__init__(project_root)
        self.scenario_config = scenario_config
        self.evaluation_results = {
            "started_at": datetime.now(),
            "completed_at": None,
            "success": False,
            "score": 0,
            "checks": {},
            "metrics": {},
            "errors": []
        }
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the current state against expectations."""
        expectations = self.scenario_config.get("expectations", {})
        evaluation = self.scenario_config.get("evaluation", {})
        
        results = {}
        
        # Check files exist
        if "files_should_exist" in expectations:
            results["all_files_exist"] = self._check_files_exist(
                expectations["files_should_exist"]
            )
        
        # Check file contents
        if "files_should_contain" in expectations:
            results["content_checks_pass"] = self._check_file_contents(
                expectations["files_should_contain"]
            )
        
        # Check files don't contain certain content
        if "files_should_not_contain" in expectations:
            results["no_forbidden_content"] = self._check_file_not_contains(
                expectations["files_should_not_contain"]
            )
        
        # Check tests pass
        if expectations.get("tests_should_pass", False):
            results["tests_pass"] = self._check_tests_pass()
        
        # Check custom expressions
        if "custom_checks" in expectations:
            results["custom_checks_pass"] = self._check_custom_expressions(
                expectations["custom_checks"]
            )
        
        # Calculate score
        score = 0
        scoring = evaluation.get("scoring", {})
        for check, passed in results.items():
            if passed and check in scoring:
                score += scoring[check]
        
        # Determine overall success
        success_criteria = evaluation.get("success_criteria", list(results.keys()))
        success = all(results.get(criterion, False) for criterion in success_criteria)
        
        self.evaluation_results.update({
            "completed_at": datetime.now(),
            "success": success,
            "score": score,
            "checks": results
        })
        
        return self.evaluation_results
    
    def _check_files_exist(self, files: List[str]) -> bool:
        """Check if all specified files exist."""
        for file_path in files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.evaluation_results["errors"].append(
                    f"File does not exist: {file_path}"
                )
                return False
        return True
    
    def _check_file_contents(self, checks: List[Dict[str, Any]]) -> bool:
        """Check if files contain expected content."""
        all_passed = True
        
        for check in checks:
            file_path = check["path"]
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                self.evaluation_results["errors"].append(
                    f"Cannot check content - file does not exist: {file_path}"
                )
                all_passed = False
                continue
            
            content = full_path.read_text()
            
            for expected in check.get("contains", []):
                if expected not in content:
                    self.evaluation_results["errors"].append(
                        f"File {file_path} does not contain: {expected}"
                    )
                    all_passed = False
        
        return all_passed
    
    def _check_file_not_contains(self, checks: List[Dict[str, Any]]) -> bool:
        """Check if files don't contain forbidden content."""
        all_passed = True
        
        for check in checks:
            file_path = check["path"]
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                continue  # File doesn't exist, so it doesn't contain forbidden content
            
            content = full_path.read_text()
            
            for forbidden in check.get("not_contains", []):
                if forbidden in content:
                    self.evaluation_results["errors"].append(
                        f"File {file_path} contains forbidden content: {forbidden}"
                    )
                    all_passed = False
        
        return all_passed
    
    def _check_tests_pass(self) -> bool:
        """Check if tests pass."""
        test_result = self.run_tests()
        return test_result.get("success", False)
    
    def _check_custom_expressions(self, checks: List[Dict[str, Any]]) -> bool:
        """Evaluate custom Python expressions."""
        all_passed = True
        
        for check in checks:
            name = check["name"]
            expression = check["expression"]
            
            try:
                # Change to project directory for evaluation
                original_dir = os.getcwd()
                os.chdir(self.project_root)
                
                # Evaluate expression
                result = eval(expression, {"__builtins__": {}}, {
                    "open": open,
                    "len": len,
                    "os": os,
                    "Path": Path
                })
                
                os.chdir(original_dir)
                
                if not result:
                    self.evaluation_results["errors"].append(
                        f"Custom check failed: {name}"
                    )
                    all_passed = False
                    
            except Exception as e:
                os.chdir(original_dir)
                self.evaluation_results["errors"].append(
                    f"Error in custom check '{name}': {str(e)}"
                )
                all_passed = False
        
        return all_passed


class EnvironmentLoader:
    """Loads test environments from YAML configuration files."""
    
    @staticmethod
    def load_scenario(yaml_path: str, base_dir: Optional[str] = None) -> TestEnvironment:
        """
        Load a test scenario from a YAML file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            base_dir: Base directory for test projects (defaults to qirk_tests/fixtures)
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if base_dir is None:
            base_dir = Path(__file__).parent / "fixtures"
        
        # Extract environment config
        env_config = config.get("environment", {})
        initial_state = env_config.get("initial_state", {})
        
        # Create project directory
        project_name = Path(yaml_path).stem
        project_root = Path(base_dir) / project_name
        
        # Clean up if exists
        if project_root.exists():
            shutil.rmtree(project_root)
        
        project_root.mkdir(parents=True, exist_ok=True)
        
        # Create initial files
        for file_info in initial_state.get("files", []):
            file_path = project_root / file_info["path"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(file_info["content"])
        
        # Initialize git if requested
        git_config = initial_state.get("git", {})
        if git_config.get("initialize", False):
            subprocess.run(["git", "init"], cwd=project_root, capture_output=True)
            
            if git_config.get("initial_commit", False):
                subprocess.run(["git", "add", "."], cwd=project_root, capture_output=True)
                commit_msg = git_config.get("commit_message", "Initial commit")
                subprocess.run(
                    ["git", "commit", "-m", commit_msg],
                    cwd=project_root,
                    capture_output=True
                )
        
        # Create test environment
        env = TestEnvironment(str(project_root), env_config)
        
        # Set initial state
        state_config = initial_state.get("state", {})
        env.update_state(state_config)
        
        return env
    
    @staticmethod
    def load_all_scenarios(scenarios_dir: str) -> Dict[str, TestEnvironment]:
        """Load all scenario YAML files from a directory."""
        scenarios = {}
        
        scenarios_path = Path(scenarios_dir)
        for yaml_file in scenarios_path.glob("*.yaml"):
            if yaml_file.name != "environment_schema.yaml":
                try:
                    env = EnvironmentLoader.load_scenario(str(yaml_file))
                    scenarios[yaml_file.stem] = env
                except Exception as e:
                    print(f"Error loading scenario {yaml_file}: {e}")
        
        return scenarios
    
    @staticmethod
    def extract_tasks_from_config(config: Dict[str, Any]) -> List[str]:
        """
        Extract initial tasks from environment config.
        This allows the agent to read the environment and figure out what to do.
        """
        # First check if there are explicit tasks
        initial_state = config.get("initial_state", {})
        state = initial_state.get("state", {})
        
        if "pending_tasks" in state and state["pending_tasks"]:
            return state["pending_tasks"]
        
        # Otherwise, infer tasks from metadata and expectations
        metadata = config.get("metadata", {})
        expectations = config.get("expectations", {})
        
        inferred_tasks = []
        
        # Infer from test expectations
        if expectations.get("tests_should_pass", False):
            test_files = [
                f["path"] for f in initial_state.get("files", [])
                if "test" in f["path"]
            ]
            if test_files:
                inferred_tasks.append(
                    f"Make sure the tests in {', '.join(test_files)} pass"
                )
        
        # Infer from content expectations
        for content_check in expectations.get("files_should_contain", []):
            path = content_check["path"]
            contains = content_check.get("contains", [])
            if contains:
                inferred_tasks.append(
                    f"Update {path} to include: {', '.join(contains)}"
                )
        
        # Infer from tags
        tags = metadata.get("tags", [])
        if "bug-fix" in tags:
            inferred_tasks.insert(0, "Find and fix the bug in the code")
        elif "feature" in tags:
            inferred_tasks.insert(0, "Implement the requested feature")
        elif "refactoring" in tags:
            inferred_tasks.insert(0, "Refactor the code for better quality")
        
        return inferred_tasks if inferred_tasks else [
            "Analyze the code and make appropriate improvements"
        ]