"""
QirkTheCoder - A state-of-the-art coding agent inspired by Claude Code, Gemini CLI, and Codex.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from core.base import Agent, LM, tool
from environments.coding_environment import CodingEnvironment
from core.template_loader import QirkPromptTemplate


class QirkTheCoder(Agent):
    """
    A sophisticated coding agent that can:
    - Read and write files with full context
    - Apply patches for precise edits
    - Run tests and fix failures
    - Manage multi-file changes
    - Handle git operations
    """
    
    def __init__(self, lm: LM, project_root: str = ".", environment: CodingEnvironment = None):
        if environment is None:
            environment = CodingEnvironment(project_root)
        super().__init__(lm, environment)
        
        self.project_root = Path(project_root).resolve()
        self.config = QirkPromptTemplate.load("config/qirkCode.yaml")
        
        # Trajectory management
        self.trajectory: List[Dict[str, Any]] = []
        self.max_trajectory_length = self.config.get("context_management", {}).get("max_trajectory_length", 50)
        self.compression_threshold = self.config.get("context_management", {}).get("compression_threshold", 30000)
        self.preserve_recent = self.config.get("context_management", {}).get("preserve_recent_messages", 10)
        
        # Add system prompt to trajectory
        system_prompt = self.config.get("system_prompts", {}).get("default", "")
        if system_prompt:
            self.trajectory.append({"role": "system", "content": system_prompt})
        
    def step(self, environment: CodingEnvironment) -> Dict[str, Any]:
        """Execute one step of coding work."""
        state = self.ingest_state(environment)
        
        # Check for pending tasks
        pending_tasks = state.get("pending_tasks", [])
        if not pending_tasks:
            return {"status": "idle", "message": "No pending tasks"}
        
        # Get current task
        current_task = pending_tasks[0]
        
        # Build context-aware messages
        messages = self._build_context_messages(current_task, state)
        
        # Invoke LM
        result = self.invoke(messages)
        
        # Update task list
        remaining_tasks = pending_tasks[1:]
        completed_tasks = state.get("completed_tasks", [])
        completed_tasks.append({
            "task": current_task,
            "completed_at": datetime.now().isoformat(),
            "result": result.get("assistant_message", {}).get("content", "")
        })
        
        environment.update_state({
            "pending_tasks": remaining_tasks,
            "completed_tasks": completed_tasks
        })
        
        return {
            "status": "completed",
            "task": current_task,
            "remaining": len(remaining_tasks)
        }
    
    def _build_context_messages(self, task: str, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build messages with trajectory management and context compression."""
        # Compress trajectory if needed
        compressed_trajectory = self._manage_trajectory()
        
        # Build current context
        context_info = f"""Current context:
- Project root: {state.get('project_root', '.')}
- Modified files: {', '.join(state.get('modified_files', [])) or 'none'}
- Test status: {state.get('test_status', 'unknown')}
- Current focus: {state.get('current_focus', 'none')}"""
        
        # Combine trajectory with current task
        messages = compressed_trajectory.copy()
        messages.append({"role": "system", "content": context_info})
        messages.append({"role": "user", "content": task})
        
        return messages
    
    def _manage_trajectory(self) -> List[Dict[str, Any]]:
        """Manage trajectory with compression and windowing."""
        # Calculate current size
        current_size = sum(len(m.get("content", "")) for m in self.trajectory)
        
        if current_size > self.compression_threshold or len(self.trajectory) > self.max_trajectory_length:
            # Compress older messages
            return self._compress_trajectory()
        
        return self.trajectory.copy()
    
    def _compress_trajectory(self) -> List[Dict[str, Any]]:
        """Compress trajectory by summarizing older interactions."""
        if len(self.trajectory) <= self.preserve_recent:
            return self.trajectory.copy()
        
        # Keep system message
        compressed = []
        if self.trajectory and self.trajectory[0].get("role") == "system":
            compressed.append(self.trajectory[0])
        
        # Summarize older messages
        older_messages = self.trajectory[1:-self.preserve_recent]
        if older_messages:
            summary = self._summarize_messages(older_messages)
            compressed.append({
                "role": "system",
                "content": f"Summary of previous interactions:\n{summary}"
            })
        
        # Keep recent messages
        compressed.extend(self.trajectory[-self.preserve_recent:])
        
        return compressed
    
    def _summarize_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize a list of messages into key points."""
        # Extract key information
        files_modified = set()
        tools_used = []
        tasks_completed = []
        
        for msg in messages:
            content = msg.get("content", "")
            
            # Extract file modifications
            if "wrote" in content or "modified" in content:
                # Simple extraction - in production, use better parsing
                import re
                files = re.findall(r"'([^']+\.\w+)'", content)
                files_modified.update(files)
            
            # Extract tool usage
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tool_call in msg.get("tool_calls", []):
                    tools_used.append(tool_call.function.name)
            
            # Extract completed tasks
            if "completed" in content or "fixed" in content:
                tasks_completed.append(content[:100] + "...")
        
        summary = "Previous work:\n"
        if files_modified:
            summary += f"- Modified files: {', '.join(sorted(files_modified))}\n"
        if tools_used:
            tool_counts = {}
            for tool in tools_used:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
            summary += f"- Tools used: {', '.join(f'{t}({c}x)' for t, c in tool_counts.items())}\n"
        if tasks_completed:
            summary += f"- Completed {len(tasks_completed)} tasks\n"
        
        return summary
    
    def invoke(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Override invoke to update trajectory."""
        result = super().invoke(messages)
        
        # Add to trajectory
        if messages:
            self.trajectory.extend(messages)
        
        if "assistant_message" in result:
            self.trajectory.append(result["assistant_message"])
        
        if "observation_message" in result:
            self.trajectory.append(result["observation_message"])
        
        return result
    
    @tool
    def read_file(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """
        Read a file with line numbers.
        
        Args:
            file_path: Path to the file relative to project root
            start_line: Optional starting line number (1-indexed)
            end_line: Optional ending line number (inclusive)
        """
        try:
            abs_path = self.project_root / file_path
            
            if not abs_path.exists():
                return f"Error: File '{file_path}' does not exist"
            
            with open(abs_path, 'r') as f:
                lines = f.readlines()
            
            # Load into environment
            env = self.environment
            env.load_file(file_path)
            
            # Apply line range if specified
            if start_line is not None:
                start_idx = max(0, start_line - 1)
                end_idx = end_line if end_line is not None else len(lines)
                lines = lines[start_idx:end_idx]
                line_offset = start_idx
            else:
                line_offset = 0
            
            # Format with line numbers
            formatted_lines = []
            for i, line in enumerate(lines):
                line_num = i + line_offset + 1
                formatted_lines.append(f"{line_num:4d} | {line.rstrip()}")
            
            content = "\n".join(formatted_lines)
            
            # Update environment focus
            env.update_state({"current_focus": file_path})
            
            return f"File: {file_path}\n{content}"
            
        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"
    
    @tool
    def write_file(self, file_path: str, content: str) -> str:
        """
        Write or create a file with the given content.
        
        Args:
            file_path: Path to the file relative to project root
            content: Complete file content
        """
        try:
            env = self.environment
            file_state = env.load_file(file_path)
            
            # Update content
            file_state.update(content, "Written by write_file tool")
            
            # Save to disk
            if env.save_file(file_path):
                return f"Successfully wrote {len(content)} characters to '{file_path}'"
            else:
                return f"Error: Failed to save '{file_path}'"
                
        except Exception as e:
            return f"Error writing file '{file_path}': {str(e)}"
    
    @tool
    def apply_patch(self, file_path: str, patch: str) -> str:
        """
        Apply a unified diff patch to a file.
        
        Args:
            file_path: Path to the file to patch
            patch: Unified diff format patch
        """
        try:
            env = self.environment
            success, message = env.apply_patch(file_path, patch)
            
            if success:
                # Run tests if configured
                if env._state.get("auto_test", False):
                    test_result = env.run_tests()
                    if not test_result["success"]:
                        return f"Patch applied but tests failed:\n{test_result.get('stdout', '')}\n{test_result.get('stderr', '')}"
                
                return message
            else:
                return f"Failed to apply patch: {message}"
                
        except Exception as e:
            return f"Error applying patch: {str(e)}"
    
    @tool
    def search_files(self, pattern: str, file_pattern: Optional[str] = None, max_results: int = 20) -> str:
        """
        Search for a pattern across files in the project.
        
        Args:
            pattern: Regex pattern to search for
            file_pattern: Optional glob pattern for files (e.g., "*.py")
            max_results: Maximum number of results to return
        """
        try:
            # Build ripgrep command
            cmd = ["rg", "--line-number", "--no-heading", "-m", str(max_results)]
            
            if file_pattern:
                cmd.extend(["-g", file_pattern])
            
            cmd.append(pattern)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                return f"Search results for '{pattern}':\n{result.stdout}"
            elif result.returncode == 1:
                return f"No matches found for '{pattern}'"
            else:
                # Fallback to grep if ripgrep not available
                cmd = ["grep", "-rn", "--include", file_pattern or "*", pattern, "."]
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    lines = result.stdout.strip().split('\n')[:max_results]
                    return f"Search results for '{pattern}':\n" + "\n".join(lines)
                else:
                    return f"No matches found for '{pattern}'"
                    
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    @tool
    def run_tests(self, test_command: Optional[str] = None) -> str:
        """
        Run the project's test suite.
        
        Args:
            test_command: Optional custom test command
        """
        try:
            env = self.environment
            result = env.run_tests(test_command)
            
            output = f"Test command: {result['command']}\n"
            output += f"Status: {'PASSED' if result['success'] else 'FAILED'}\n\n"
            
            if result.get('stdout'):
                output += f"Output:\n{result['stdout']}\n"
            
            if result.get('stderr'):
                output += f"Errors:\n{result['stderr']}\n"
            
            if result.get('error'):
                output += f"Error: {result['error']}\n"
            
            return output
            
        except Exception as e:
            return f"Error running tests: {str(e)}"
    
    @tool
    def run_command(self, command: str, timeout: int = 30) -> str:
        """
        Run a shell command in the project directory.
        
        Args:
            command: Shell command to execute
            timeout: Timeout in seconds
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = f"Command: {command}\n"
            output += f"Exit code: {result.returncode}\n\n"
            
            if result.stdout:
                output += f"Output:\n{result.stdout}\n"
            
            if result.stderr:
                output += f"Errors:\n{result.stderr}\n"
            
            return output
            
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error running command: {str(e)}"
    
    @tool
    def view_project_structure(self, max_depth: int = 3) -> str:
        """
        View the project directory structure.
        
        Args:
            max_depth: Maximum depth to traverse
        """
        try:
            env = self.environment
            tree = env.get_project_structure(max_depth)
            
            def format_tree(node, prefix="", is_last=True):
                if not node:
                    return ""
                
                result = prefix
                result += "└── " if is_last else "├── "
                result += node["name"] + "\n"
                
                if node.get("type") == "directory" and node.get("children"):
                    extension = "    " if is_last else "│   "
                    children = node["children"]
                    for i, child in enumerate(children):
                        result += format_tree(
                            child,
                            prefix + extension,
                            i == len(children) - 1
                        )
                
                return result
            
            return f"Project structure:\n{format_tree(tree)}"
            
        except Exception as e:
            return f"Error viewing project structure: {str(e)}"
    
    @tool
    def create_patch_from_edit(self, file_path: str, old_content: str, new_content: str) -> str:
        """
        Create a unified diff patch from old and new content.
        
        Args:
            file_path: Path to the file
            old_content: Original content
            new_content: Modified content
        """
        try:
            env = self.environment
            patch = env.create_patch(file_path, old_content, new_content)
            
            if patch:
                return f"Created patch for {file_path}:\n{patch}"
            else:
                return "No changes detected"
                
        except Exception as e:
            return f"Error creating patch: {str(e)}"
    
    @tool
    def git_status(self) -> str:
        """Check git status of the project."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Error: Not a git repository or git not available"
            
            if not result.stdout:
                return "Working directory clean - no changes"
            
            # Parse status
            lines = result.stdout.strip().split('\n')
            modified = []
            added = []
            deleted = []
            untracked = []
            
            for line in lines:
                if line.startswith(" M"):
                    modified.append(line[3:])
                elif line.startswith("M "):
                    modified.append(line[3:] + " (staged)")
                elif line.startswith("A "):
                    added.append(line[3:])
                elif line.startswith("D "):
                    deleted.append(line[3:])
                elif line.startswith("??"):
                    untracked.append(line[3:])
            
            status = "Git status:\n"
            if modified:
                status += f"Modified: {', '.join(modified)}\n"
            if added:
                status += f"Added: {', '.join(added)}\n"
            if deleted:
                status += f"Deleted: {', '.join(deleted)}\n"
            if untracked:
                status += f"Untracked: {', '.join(untracked)}\n"
            
            return status
            
        except Exception as e:
            return f"Error checking git status: {str(e)}"
    
    @tool
    def git_diff(self, file_path: Optional[str] = None, staged: bool = False) -> str:
        """
        Show git diff for a file or all changes.
        
        Args:
            file_path: Optional specific file to diff
            staged: Show staged changes instead of working directory
        """
        try:
            cmd = ["git", "diff"]
            if staged:
                cmd.append("--cached")
            if file_path:
                cmd.append(file_path)
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            
            if not result.stdout:
                return "No differences found"
            
            return f"Git diff:\n{result.stdout}"
            
        except Exception as e:
            return f"Error getting git diff: {str(e)}"
    
    @tool
    def git_add(self, file_paths: List[str]) -> str:
        """
        Stage files for commit.
        
        Args:
            file_paths: List of file paths to stage
        """
        try:
            cmd = ["git", "add"] + file_paths
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Error staging files: {result.stderr}"
            
            return f"Staged {len(file_paths)} file(s): {', '.join(file_paths)}"
            
        except Exception as e:
            return f"Error staging files: {str(e)}"
    
    @tool
    def git_commit(self, message: str, description: Optional[str] = None) -> str:
        """
        Create a git commit with the staged changes.
        
        Args:
            message: Commit message (first line)
            description: Optional extended description
        """
        try:
            # Build commit message
            full_message = message
            if description:
                full_message += f"\n\n{description}"
            
            # Add footer from config
            commit_template = self.config.get("templates", {}).get("commit_message", "")
            if commit_template and "Generated by QirkTheCoder" in commit_template:
                full_message += "\n\nGenerated by QirkTheCoder"
            
            result = subprocess.run(
                ["git", "commit", "-m", full_message],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                if "nothing to commit" in result.stdout:
                    return "Nothing to commit - working directory clean"
                return f"Error committing: {result.stderr}"
            
            # Get commit hash
            hash_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            commit_hash = hash_result.stdout.strip()[:7] if hash_result.returncode == 0 else "unknown"
            
            return f"Created commit {commit_hash}: {message}"
            
        except Exception as e:
            return f"Error creating commit: {str(e)}"
    
    @tool
    def git_log(self, max_entries: int = 10) -> str:
        """
        Show recent git commits.
        
        Args:
            max_entries: Maximum number of commits to show
        """
        try:
            result = subprocess.run(
                ["git", "log", f"-{max_entries}", "--oneline", "--decorate"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            
            if not result.stdout:
                return "No commits found"
            
            return f"Recent commits:\n{result.stdout}"
            
        except Exception as e:
            return f"Error getting git log: {str(e)}"