"""
This module defines the CodeEditor agent.
"""

import subprocess
from typing import Any, Dict, List
from core.base import Agent, LM, LiteLLM, tool
from core.environment import Environment, SharedEnvironment
from core.template_loader import QirkPromptTemplate

class CodeEditor(Agent):
    """
    An agent that can view and modify a file in its environment.
    """

    def __init__(self, lm: LM, file_content: str, environment: Environment = None):
        super().__init__(lm, environment)
        self.environment.update_state({"file_content": file_content}, agent=self)
    
    def step(self, environment: Environment) -> Dict[str, Any]:
        """
        Execute one step of code editing.
        """
        state = self.ingest_state(environment)
        current_content = state.get("file_content", "")
        edit_requests = state.get("edit_requests", [])
        
        if not edit_requests:
            return {"status": "idle", "message": "No edit requests pending"}
        
        # Process the first edit request
        edit_request = edit_requests[0]
        messages = [{
            "role": "user",
            "content": f"Edit the file according to: {edit_request}"
        }]
        
        result = self.invoke(messages)
        
        # Remove processed request
        remaining_requests = edit_requests[1:]
        environment.update_state({"edit_requests": remaining_requests}, agent=self)
        
        return {
            "status": "edited",
            "request": edit_request,
            "remaining": len(remaining_requests)
        }

    @tool
    def update_file(self, new_content: str):
        """
        Updates the content of the file in the environment.

        Args:
            new_content: The new, complete content of the file.
        """
        self.environment.update_state({"file_content": new_content}, agent=self)
        return "File content updated successfully."

    @tool
    def view_file(self):
        """
        Views the current content of the file in the environment.
        """
        state = self.environment.get_state(self)
        return state.get("file_content", "File is empty.")


# Note: QirkTheCoder has been moved to agents/qirk_coder.py with enhanced capabilities
# This import is kept for backward compatibility
try:
    from .qirk_coder import QirkTheCoder
except ImportError:
    # Fallback implementation for compatibility
    class QirkTheCoder(Agent):
        """
        Legacy QirkTheCoder implementation.
        Please use agents.qirk_coder.QirkTheCoder for the enhanced version.
        """

        def __init__(self, lm: LiteLLM, environment: Environment = None):
            super().__init__(lm, environment)
            self.configuration = QirkPromptTemplate.load("../qirkCode.yaml")
        
        def step(self, environment: Environment) -> Dict[str, Any]:
            """
            Execute one step of the Qirk coder workflow.
            """
            state = self.ingest_state(environment)
            tasks = state.get("coding_tasks", [])
            
            if not tasks:
                return {"status": "idle", "message": "No coding tasks"}
            
            current_task = tasks[0]
            trajectory = state.get("trajectory", [])
            
            # Process task with trajectory
            result = self.invoke(trajectory + [{
                "role": "user",
                "content": current_task
            }])
            
            # Update state
            remaining_tasks = tasks[1:]
            trajectory.append(result.get("assistant_message", {}))
            
            environment.update_state({
                "coding_tasks": remaining_tasks,
                "trajectory": trajectory
            }, agent=self)
            
            return {
                "status": "processed",
                "task": current_task,
                "remaining": len(remaining_tasks)
            }

        def invoke(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
            """
            The main entry point for the agent.
            """
            return super().invoke(trajectory)

        @staticmethod
        def handle_trajectory(trajectory: List[Dict[str, Any]]):
            """
            Handles the trajectory of the agent. The trajectory of Qirk is the list of messages that have been sent to the agent.
            This truncates the trajectory if necessary and returns a new trajectory.
            NOTE: the trajectory includes the `system_message` at the beginning. index 0 should not be edited (?).
            """
            pass

        @tool
        def update_file(self, file_path: str, new_content: str):
            """
            Writes the new content to the file.
            """
            try:
                with open(file_path, 'w') as f:
                    f.write(new_content)
                return f"File '{file_path}' updated successfully."
            except Exception as e:
                return f"Error updating file '{file_path}': {e}"
        
        @tool
        def run_bash_command(self, command: str):
            """
            Runs a bash command and returns the output.
            """
            return subprocess.run(command, shell=True, capture_output=True, text=True).stdout
    