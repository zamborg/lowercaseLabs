"""
This module contains the main execution loop for running an agent.
"""


import os
from typing import Callable, List, Dict, Any
from litellm import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from core.base import Agent, LiteLLM
from agents.code_editor import CodeEditor, QirkTheCoder
from agents.preference_mapper import PreferenceMapper

class TokenUsage(BaseModel):
    """
    Represents the token usage of an agent.
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0

    def add_token_usage(self, token_usage: "TokenUsage"):
        self.prompt_tokens += token_usage.prompt_tokens
        self.completion_tokens += token_usage.completion_tokens
        self.total_tokens += token_usage.total_tokens
        self.thinking_tokens += token_usage.thinking_tokens

    def add_token_usage_raw(self, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int = 0, thinking_tokens: int = 0):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.thinking_tokens += thinking_tokens

class BaseAgentHandler:
    """
    Base class for all agent handlers.
    """
    def __init__(self, agent: Agent, max_steps: int = 20):
        self.agent = agent
        self.max_steps = max_steps
        self.console = Console()
        self.token_usage = TokenUsage()

    def run(self, initial_prompt: str):
        """
        Runs the agent loop until the task is complete or max_steps are reached.
        """
        pass

    def _run_loop(self, functor: Callable[[], Any], max_steps: int = 20):
        """
        Runs the loop until the task is complete or max_steps are reached.
        """
        if not hasattr(self.agent, "is_finished"):
            raise ValueError(f"Agent does not have an is_finished attribute, early stopping is not possible. Running for {max_steps=}")

        step = 0
        while step < max_steps:
            functor() # functor is responsible for any printing etc.
            step += 1
            if hasattr(self.agent, "is_finished") and self.agent.is_finished:
                self.console.print("[bold green]Agent has finished the task.[/bold green]")
                break

    def _print_message(self, title: str, message: str, style: str):
        """Prints a message to the console in a styled panel."""
        panel = Panel(message, title=title, border_style=style, expand=False)
        self.console.print(panel)


class AgentHandler:
    """
    Manages the execution of an agent in a loop.
    """

    def __init__(self, agent: Agent, max_steps: int = 20, predefined_answers: List[str] = None):
        self.agent = agent
        self.max_steps = max_steps
        self.console = Console()
        self.token_count = 0
        self.predefined_answers = predefined_answers or []
        self.answer_index = 0

        # Monkey-patch the agent's ask_user method to simulate user input
        if hasattr(self.agent, 'ask_user') and self.predefined_answers:
            original_ask_user = self.agent.ask_user

            def simulated_ask_user(question: str) -> str:
                self.console.print(Panel(question, title="Agent Question", border_style="cyan"))
                if self.answer_index < len(self.predefined_answers):
                    answer = self.predefined_answers[self.answer_index]
                    self.console.print(Panel(answer, title="Simulated User Answer", border_style="yellow"))
                    self.answer_index += 1
                    return answer
                return "No further answers provided."

            self.agent.ask_user = simulated_ask_user

            # Also update the tools dictionary to use the simulated function
            if 'ask_user' in self.agent.tools:
                self.agent.tools['ask_user'] = self.agent.ask_user

    def _print_message(self, title: str, message: str, style: str):
        """Prints a message to the console in a styled panel."""
        panel = Panel(message, title=title, border_style=style, expand=False)
        self.console.print(panel)

    def run(self, initial_prompt: str):
        """
        Runs the agent loop until the task is complete or max_steps are reached.
        """
        messages: List[Dict[str, Any]] = [{"role": "user", "content": initial_prompt}]
        step = 0

        while step < self.max_steps:
            self.console.rule(f"[bold cyan]Step {step + 1}")

            # Let the agent decide on the next action
            result = self.agent.invoke(messages)

            # Add the assistant's response to the history
            assistant_message = result["assistant_message"]
            messages.append(assistant_message)

            if not assistant_message.tool_calls:
                self._print_message("Assistant Response", assistant_message.content, "green")
                # If the agent responds without a tool call, we treat it as a message
                # to the user and continue the loop, waiting for the next user input.
                # In our simulated case, the next answer will be fed in the next iteration.
                step += 1
                continue

            # Print the tool call and add the observation to history
            tool_name = assistant_message.tool_calls[0].function.name
            tool_args = assistant_message.tool_calls[0].function.arguments
            self._print_message("Tool Call", f"{tool_name}({tool_args})", "bold blue")

            observation_message = result["observation_message"]
            messages.append(observation_message)
            self._print_message("Observation", str(observation_message["content"]), "magenta")

            # Check for termination condition
            if hasattr(self.agent, "is_finished") and self.agent.is_finished:
                self.console.print("[bold green]Agent has finished the task.[/bold green]")
                break

            step += 1

        if step >= self.max_steps:
            self.console.print("[bold red]Max steps reached. Halting execution.[/bold red]")

        # Optional: Display final state
        if isinstance(self.agent, CodeEditor):
            final_code = self.agent.view_file()
            self.console.print(Panel(
                Syntax(final_code, "python", theme="monokai", line_numbers=True),
                title="Final Code",
                border_style="green"
            ))

class QirkHandler(BaseAgentHandler):
    """
    Handler for the Qirk agent.
    """
    def __init__(self, agent: QirkTheCoder, max_steps: int = 5):
        super().__init__(agent, max_steps)
        self.trajectory: List[Dict[str, Any]] = []

    def run(self, task: str):
        """
        Runs the agent loop until the task is complete or max_steps are reached.
        """
        step = 0
        while step < self.max_steps:
            self.console.rule(f"[bold cyan]Step {step}")
            result = self.agent.invoke(task)
            assistant_message = result["assistant_message"]
            self.trajectory.append(assistant_message)
            self._print_message("Assistant Response", assistant_message.content, "green")
            step += 1
            


if __name__ == "__main__":
    # 1. Set up the agent
    lm_instance = LiteLLM(model="gpt-4-turbo")
    # preference_agent = PreferenceMapper(lm=lm_instance)

    # # 2. Define the task and simulated user answers
    # task_prompt = (
    #     "Your job is to interview me to understand my preferences for analyzing a meeting transcript. "
    #     "You need to find out what I want to focus on, what format I want the output in, and any key topics I care about. "
    #     "Ask me questions one at a time. Once you have a clear picture, synthesize my preferences into a detailed prompt that another LLM can use. "
    #     "Save this prompt to a markdown file and then exit."
    # )
    # simulated_answers = [
    #     "I want to focus on action items and the key decisions made.",
    #     "Output the analysis as a bulleted list inside a markdown file.",
    #     "Yes, please track the contributions of a person named 'Alex'.",
    #     "No, that's all the information I need. Please proceed with saving the preferences."
    # ]

    # # 3. Run the handler
    # handler = AgentHandler(agent=preference_agent, predefined_answers=simulated_answers)
    # handler.run(initial_prompt=task_prompt)


    # Example for CodeEditor (uncomment to test)
    print("\n" + "="*50 + "\n")
    print("Testing CodeEditor Agent")
    initial_code = "def hello_world():\n    print('Hello, world!')"
    code_editor_agent = CodeEditor(lm=lm_instance, file_content=initial_code)
    code_editor_handler = AgentHandler(agent=code_editor_agent)
    task = input("what would you like to do?\n")
    code_editor_handler.run(initial_prompt=task)