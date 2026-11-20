"""
FileSystemEnvironment for grep-based code search tasks.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from core.environment import Environment


class FileSystemEnvironment(Environment):
    """
    Environment representing a file system with a git repository checked out.
    Designed for agents that need to search for code elements using bash commands.
    """
    
    def __init__(self, repo_url: str = None, commit_hash: str = None, local_path: str = None):
        """
        Initialize file system environment.
        
        Args:
            repo_url: Git repository URL to clone
            commit_hash: Specific commit to checkout
            local_path: Use existing local repository instead of cloning
        """
        super().__init__()
        
        if local_path:
            self.repo_path = Path(local_path).resolve()
            if not self.repo_path.exists():
                raise ValueError(f"Local path does not exist: {local_path}")
        else:
            # Create temporary directory for repo
            self.temp_dir = tempfile.mkdtemp(prefix="grep_agent_")
            self.repo_path = Path(self.temp_dir)
            
            if repo_url:
                self._clone_repo(repo_url, commit_hash)
        
        # Initialize state
        self._state = {
            "repo_url": repo_url,
            "commit_hash": commit_hash,
            "repo_path": str(self.repo_path),
            "search_history": [],
            "commands_executed": [],
            "current_directory": str(self.repo_path),
            "found_matches": [],
            "start_time": datetime.now().isoformat()
        }
    
    def _clone_repo(self, repo_url: str, commit_hash: Optional[str] = None):
        """Clone repository and optionally checkout specific commit."""
        try:
            # Clone repository
            result = subprocess.run(
                ["git", "clone", repo_url, str(self.repo_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to clone repo: {result.stderr}")
            
            # Checkout specific commit if provided
            if commit_hash:
                result = subprocess.run(
                    ["git", "checkout", commit_hash],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to checkout commit: {result.stderr}")
                    
        except Exception as e:
            self.cleanup()
            raise e
    
    def get_state(self, agent: Optional["Agent"] = None) -> Dict[str, Any]:
        """Get current environment state."""
        state = self._state.copy()
        
        # Add summary statistics
        state["total_commands"] = len(state["commands_executed"])
        state["total_matches"] = len(state["found_matches"])
        
        return state
    
    def update_state(self, updates: Dict[str, Any], agent: Optional["Agent"] = None):
        """Update environment state."""
        self._state.update(updates)
    
    def execute_command(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """
        Execute a bash command in the repository directory.
        
        Args:
            command: Bash command to execute
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (success, stdout, stderr)
        """
        # Record command
        self._state["commands_executed"].append({
            "command": command,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            # Execute command in repo directory
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            stdout = result.stdout
            stderr = result.stderr
            
            # Record result
            self._state["commands_executed"][-1].update({
                "success": success,
                "stdout_lines": len(stdout.splitlines()) if stdout else 0,
                "stderr_lines": len(stderr.splitlines()) if stderr else 0
            })
            
            return success, stdout, stderr
            
        except subprocess.TimeoutExpired:
            self._state["commands_executed"][-1]["timeout"] = True
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            self._state["commands_executed"][-1]["error"] = str(e)
            return False, "", str(e)
    
    def record_match(self, filepath: str, line_number: int = None, confidence: float = 1.0):
        """Record a potential match found by the agent."""
        match = {
            "filepath": filepath,
            "line_number": line_number,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        self._state["found_matches"].append(match)
    
    def get_file_content(self, filepath: str) -> Optional[str]:
        """Get content of a file in the repository."""
        try:
            full_path = self.repo_path / filepath
            if full_path.exists() and full_path.is_file():
                return full_path.read_text()
            return None
        except Exception:
            return None
    
    def cleanup(self):
        """Clean up temporary directory if created."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()


class GrepTaskEnvironment(FileSystemEnvironment):
    """
    Specialized environment for grep-based function finding tasks.
    """
    
    def __init__(self, repo_url: str, commit_hash: str, function_name: str, 
                 expected_filepath: str = None, local_path: str = None):
        """
        Initialize grep task environment.
        
        Args:
            repo_url: Git repository URL
            commit_hash: Specific commit hash
            function_name: Function to search for
            expected_filepath: Expected file containing the function (for evaluation)
            local_path: Use existing local repo instead of cloning
        """
        super().__init__(repo_url, commit_hash, local_path)
        
        # Task-specific state
        self._state.update({
            "function_name": function_name,
            "expected_filepath": expected_filepath,
            "task_completed": False,
            "final_answer": None
        })
    
    def submit_answer(self, filepath: str, line_number: int = None):
        """Submit final answer for the grep task."""
        self._state["final_answer"] = {
            "filepath": filepath,
            "line_number": line_number,
            "timestamp": datetime.now().isoformat()
        }
        self._state["task_completed"] = True
        
        # Record as match
        self.record_match(filepath, line_number, confidence=1.0)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the agent's performance."""
        if not self._state["task_completed"]:
            return {
                "success": False,
                "reason": "Task not completed",
                "score": 0.0
            }
        
        final_answer = self._state["final_answer"]
        if not final_answer:
            return {
                "success": False,
                "reason": "No answer submitted",
                "score": 0.0
            }
        
        # Check if answer matches expected
        submitted_path = final_answer["filepath"]
        expected_path = self._state.get("expected_filepath")
        
        if not expected_path:
            # Can't evaluate without ground truth
            return {
                "success": True,
                "reason": "No ground truth available",
                "score": None,
                "submitted": submitted_path
            }
        
        # Normalize paths for comparison
        submitted_normalized = Path(submitted_path).as_posix()
        expected_normalized = Path(expected_path).as_posix()
        
        # Check exact match
        exact_match = submitted_normalized == expected_normalized
        
        # Check if same file (might have different path representations)
        same_file = submitted_normalized.endswith(expected_normalized) or \
                   expected_normalized.endswith(submitted_normalized)
        
        # Calculate metrics
        total_commands = len(self._state["commands_executed"])
        grep_commands = sum(1 for cmd in self._state["commands_executed"] 
                          if "grep" in cmd["command"])
        
        return {
            "success": exact_match or same_file,
            "exact_match": exact_match,
            "same_file": same_file,
            "submitted": submitted_normalized,
            "expected": expected_normalized,
            "score": 1.0 if exact_match else (0.8 if same_file else 0.0),
            "metrics": {
                "total_commands": total_commands,
                "grep_commands": grep_commands,
                "efficiency": 1.0 / (total_commands + 1),  # Fewer commands = better
                "duration_seconds": (
                    datetime.now() - datetime.fromisoformat(self._state["start_time"])
                ).total_seconds()
            }
        }