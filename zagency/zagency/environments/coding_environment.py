"""
CodingEnvironment - A specialized environment for code editing agents.
"""

import os
import difflib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from zagency.core.environment import Environment


class FileState:
    """Tracks the state of a file including content and modifications."""
    
    def __init__(self, path: str, content: str = ""):
        self.path = path
        self.original_content = content
        self.current_content = content
        self.history: List[Dict[str, Any]] = []
        self.last_modified = datetime.now()
    
    def update(self, new_content: str, change_description: str = ""):
        """Update file content and track the change."""
        self.history.append({
            "timestamp": datetime.now(),
            "previous_content": self.current_content,
            "new_content": new_content,
            "description": change_description
        })
        self.current_content = new_content
        self.last_modified = datetime.now()
    
    def get_diff(self) -> str:
        """Get unified diff between original and current content."""
        original_lines = self.original_content.splitlines(keepends=True)
        current_lines = self.current_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            current_lines,
            fromfile=f"{self.path} (original)",
            tofile=f"{self.path} (current)",
            lineterm=""
        )
        return "".join(diff)
    
    def rollback(self, steps: int = 1):
        """Rollback to a previous state."""
        actual_steps = min(steps, len(self.history))
        for _ in range(actual_steps):
            if self.history:
                prev_state = self.history.pop()
                self.current_content = prev_state["previous_content"]
        self.last_modified = datetime.now()


class CodingEnvironment(Environment):
    """
    Environment specifically designed for code editing agents.
    Manages file states, test results, and code transformations.
    """
    
    def __init__(self, project_root: str = "."):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.file_states: Dict[str, FileState] = {}
        self.test_results: List[Dict[str, Any]] = []
        self.pending_patches: List[Dict[str, Any]] = []
        
        # Initialize state
        self._state = {
            "project_root": str(self.project_root),
            "modified_files": [],
            "test_status": "unknown",
            "last_test_output": "",
            "pending_tasks": [],
            "completed_tasks": [],
            "current_focus": None,
            "error_log": []
        }
    
    def get_state(self, agent: Optional["Agent"] = None) -> Dict[str, Any]:
        """Get the current environment state."""
        state = self._state.copy()
        
        # Add file modification summary
        state["file_summary"] = {
            path: {
                "modified": fs.current_content != fs.original_content,
                "changes": len(fs.history),
                "last_modified": fs.last_modified.isoformat()
            }
            for path, fs in self.file_states.items()
        }
        
        # Add pending patches count
        state["pending_patches_count"] = len(self.pending_patches)
        
        return state
    
    def update_state(self, updates: Dict[str, Any], agent: Optional["Agent"] = None):
        """Update the environment state."""
        self._state.update(updates)
    
    def load_file(self, file_path: str) -> FileState:
        """Load a file into the environment."""
        abs_path = (self.project_root / file_path).resolve()
        
        # Security check - ensure file is within project root
        if not str(abs_path).startswith(str(self.project_root)):
            raise ValueError(f"File path {file_path} is outside project root")
        
        if file_path not in self.file_states:
            if abs_path.exists():
                with open(abs_path, 'r') as f:
                    content = f.read()
            else:
                content = ""
            
            self.file_states[file_path] = FileState(file_path, content)
        
        return self.file_states[file_path]
    
    def save_file(self, file_path: str) -> bool:
        """Save a file's current state to disk."""
        if file_path not in self.file_states:
            return False
        
        file_state = self.file_states[file_path]
        abs_path = self.project_root / file_path
        
        # Create directory if needed
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(abs_path, 'w') as f:
            f.write(file_state.current_content)
        
        # Update modified files list
        if file_path not in self._state["modified_files"]:
            self._state["modified_files"].append(file_path)
        
        return True
    
    def create_patch(self, file_path: str, original: str, modified: str) -> str:
        """Create a unified diff patch."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=file_path,
            tofile=file_path,
            lineterm=""
        )
        return "".join(diff)
    
    def apply_patch(self, file_path: str, patch: str) -> Tuple[bool, str]:
        """Apply a patch to a file."""
        file_state = self.load_file(file_path)
        
        # For now, we'll use a simple approach
        # In production, you'd use a proper patch library
        try:
            # Save patch to temp file
            patch_file = self.project_root / f".patch_{datetime.now().timestamp()}"
            with open(patch_file, 'w') as f:
                f.write(patch)
            
            # Apply patch using system patch command
            target_file = self.project_root / file_path
            result = subprocess.run(
                ["patch", str(target_file), str(patch_file)],
                capture_output=True,
                text=True
            )
            
            # Clean up
            patch_file.unlink()
            
            if result.returncode == 0:
                # Reload file content
                with open(target_file, 'r') as f:
                    new_content = f.read()
                file_state.update(new_content, "Applied patch")
                return True, "Patch applied successfully"
            else:
                return False, f"Patch failed: {result.stderr}"
                
        except Exception as e:
            return False, f"Error applying patch: {str(e)}"
    
    def run_tests(self, test_command: str = None) -> Dict[str, Any]:
        """Run tests and capture results."""
        if not test_command:
            # Try to detect test command
            if (self.project_root / "package.json").exists():
                test_command = "npm test"
            elif (self.project_root / "pytest.ini").exists() or (self.project_root / "tests").exists():
                test_command = "pytest"
            elif (self.project_root / "Cargo.toml").exists():
                test_command = "cargo test"
            else:
                test_command = "make test"  # Fallback
        
        try:
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            test_result = {
                "command": test_command,
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }
            
            self.test_results.append(test_result)
            self._state["test_status"] = "passed" if test_result["success"] else "failed"
            self._state["last_test_output"] = test_result["stdout"] + test_result["stderr"]
            
            return test_result
            
        except subprocess.TimeoutExpired:
            test_result = {
                "command": test_command,
                "success": False,
                "error": "Test timed out after 60 seconds",
                "timestamp": datetime.now().isoformat()
            }
            self.test_results.append(test_result)
            self._state["test_status"] = "timeout"
            return test_result
        except Exception as e:
            test_result = {
                "command": test_command,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.test_results.append(test_result)
            self._state["test_status"] = "error"
            return test_result
    
    def get_project_structure(self, max_depth: int = 3) -> Dict[str, Any]:
        """Get a tree structure of the project."""
        def build_tree(path: Path, depth: int = 0):
            if depth > max_depth:
                return None
            
            if path.is_file():
                return {"type": "file", "name": path.name}
            elif path.is_dir():
                # Skip common ignored directories
                if path.name in ['.git', '__pycache__', 'node_modules', '.env', '.venv']:
                    return None
                
                children = []
                for child in sorted(path.iterdir()):
                    child_tree = build_tree(child, depth + 1)
                    if child_tree:
                        children.append(child_tree)
                
                return {
                    "type": "directory",
                    "name": path.name,
                    "children": children
                }
        
        return build_tree(self.project_root)