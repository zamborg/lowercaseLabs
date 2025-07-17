"""
This module defines the CodeEditor agent.
"""

import subprocess
from typing import Any, Dict, List
from core.base import Agent, LM, LiteLLM, tool
from core.template_loader import QirkPromptTemplate

class CodeEditor(Agent):
    """
    An agent that can view and modify a file in its environment.
    """

    def __init__(self, lm: LM, file_content: str):
        super().__init__(lm)
        self._environment["file_content"] = file_content

    @tool
    def update_file(self, new_content: str):
        """
        Updates the content of the file in the environment.

        Args:
            new_content: The new, complete content of the file.
        """
        self._environment["file_content"] = new_content
        return "File content updated successfully."

    @tool
    def view_file(self):
        """
        Views the current content of the file in the environment.
        """
        return self.environment.get("file_content", "File is empty.")


class QirkTheCoder(Agent):
    """
    A code editor agent that can view and modify a file in its environment.
    """

    def __init__(self, lm: LiteLLM):
        super().__init__(lm)
        self.configuration = QirkPromptTemplate.load("../qirkCode.yaml")

    def invoke(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        The main entry point for the agent.
        """
        

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
    