"""
This module defines the PreferenceMapper agent.
"""

import os
from typing import Dict, Any
from core.base import Agent, LM, tool

class PreferenceMapper(Agent):
    """
    An agent that interviews a user to determine their preferences for meeting analysis,
    then saves those preferences to a markdown file.
    """

    def __init__(self, lm: LM):
        super().__init__(lm)
        self.is_finished = False

    @tool
    def ask_user(self, question: str) -> str:
        """
        Asks the user a question to gather more information about their preferences.
        Use this to clarify requirements before saving the final preferences.

        Args:
            question: The question to ask the user.

        Returns:
            The user's answer to the question.
        """
        # In a real application, this would involve a user input loop.
        # For this simulation, we'll print the question and get input from the console.
        print(f"[Agent asks]: {question}")
        return input("[Your answer]: ")

    @tool
    def save_preferences(self, preferences: str, filename: str = "meeting_analysis_preferences.md") -> str:
        """
        Saves the summarized user preferences to a markdown file in the ./templates directory.

        Args:
            preferences: A detailed summary of the user's preferences for the final prompt.
            filename: The name of the file to save. Defaults to 'meeting_analysis_preferences.md'.
        """
        templates_dir = "templates"
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
        
        file_path = os.path.join(templates_dir, filename)
        with open(file_path, "w") as f:
            f.write(preferences)
        
        return f"Preferences saved successfully to {file_path}"

    @tool
    def exit(self):
        """
        Stops the agent's execution loop. Call this when the task is complete.
        """
        self.is_finished = True
        return "Agent execution has been stopped."
