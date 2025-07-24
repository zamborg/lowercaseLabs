"""
Step-based handler for the new agent framework.
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from zagency.core.base import Agent
from zagency.core.environment import Environment, SharedEnvironment


class StepHandler:
    """
    Handles step-based execution of agents with shared environments.
    """
    
    def __init__(self, environment: Optional[Environment] = None):
        self.environment = environment or SharedEnvironment()
        self.agents: List[Agent] = []
        self.console = Console()
        self.step_count = 0
    
    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the handler."""
        if agent not in self.agents:
            self.agents.append(agent)
            # Make sure agent uses our environment
            agent.environment = self.environment
            self.environment.register_agent(agent)
    
    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the handler."""
        if agent in self.agents:
            self.agents.remove(agent)
            self.environment.unregister_agent(agent)
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one step for all agents.
        Returns a summary of what happened in this step.
        """
        self.step_count += 1
        self.console.rule(f"[bold cyan]Step {self.step_count}")
        
        results = {}
        
        for i, agent in enumerate(self.agents):
            agent_name = f"{agent.__class__.__name__}_{i}"
            
            # Execute agent's step
            try:
                result = agent.step(self.environment)
                results[agent_name] = result
                
                # Display result
                self._display_agent_result(agent_name, result)
                
            except Exception as e:
                error_msg = f"Error in {agent_name}: {str(e)}"
                results[agent_name] = {"status": "error", "error": str(e)}
                self.console.print(f"[red]{error_msg}[/red]")
        
        # Display environment state if needed
        self._display_environment_state()
        
        return results
    
    def run(self, max_steps: int = 100, stop_condition: Optional[callable] = None):
        """
        Run the step loop until stop condition is met or max steps reached.
        
        Args:
            max_steps: Maximum number of steps to execute
            stop_condition: Optional function that takes environment state and returns True to stop
        """
        for step in range(max_steps):
            results = self.step()
            
            # Check if all agents are idle/finished
            all_idle = all(
                r.get("status") in ["idle", "finished", "waiting"] 
                for r in results.values()
            )
            
            if all_idle:
                self.console.print("[yellow]All agents are idle. Stopping execution.[/yellow]")
                break
            
            # Check custom stop condition
            if stop_condition and stop_condition(self.environment.get_state()):
                self.console.print("[green]Stop condition met. Halting execution.[/green]")
                break
        
        else:
            self.console.print(f"[red]Maximum steps ({max_steps}) reached.[/red]")
        
        self._display_final_summary()
    
    def _display_agent_result(self, agent_name: str, result: Dict[str, Any]):
        """Display the result of an agent's step."""
        status = result.get("status", "unknown")
        
        # Color based on status
        color_map = {
            "idle": "yellow",
            "waiting": "yellow", 
            "processed": "green",
            "completed": "green",
            "error": "red"
        }
        color = color_map.get(status, "white")
        
        # Create summary
        summary_parts = [f"Status: {status}"]
        for key, value in result.items():
            if key != "status":
                summary_parts.append(f"{key}: {value}")
        
        summary = " | ".join(summary_parts)
        
        panel = Panel(
            summary,
            title=f"[{color}]{agent_name}[/{color}]",
            border_style=color,
            expand=False
        )
        self.console.print(panel)
    
    def _display_environment_state(self):
        """Display current environment state in a formatted table."""
        state = self.environment.get_state()
        
        if not state:
            return
        
        # Create a table for state display
        table = Table(title="Environment State", show_header=True)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")
        
        for key, value in state.items():
            # Format value for display
            if isinstance(value, list):
                value_str = f"List[{len(value)} items]"
            elif isinstance(value, dict):
                value_str = f"Dict[{len(value)} keys]"
            else:
                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            
            table.add_row(key, value_str)
        
        self.console.print(table)
    
    def _display_final_summary(self):
        """Display a final summary of the execution."""
        self.console.rule("[bold green]Execution Complete")
        
        state = self.environment.get_state()
        
        # Show key metrics
        metrics = {
            "Total Steps": self.step_count,
            "Active Agents": len(self.agents),
        }
        
        # Add custom metrics from state
        if "processed_files" in state:
            metrics["Files Processed"] = len(state["processed_files"])
        if "completed_tasks" in state:
            metrics["Tasks Completed"] = len(state["completed_tasks"])
        
        for key, value in metrics.items():
            self.console.print(f"[bold]{key}:[/bold] {value}")


class MultiAgentOrchestrator:
    """
    Orchestrates multiple agents working together on complex tasks.
    """
    
    def __init__(self, environment: Optional[Environment] = None):
        self.handler = StepHandler(environment)
        self.console = Console()
    
    def add_agent(self, agent: Agent, role: str = None):
        """Add an agent with an optional role description."""
        self.handler.add_agent(agent)
        if role:
            self.console.print(f"[green]Added {agent.__class__.__name__} as {role}[/green]")
    
    def run_collaboration(self, initial_state: Dict[str, Any], max_steps: int = 100):
        """
        Run a collaborative session with initial state setup.
        """
        # Initialize environment with state
        self.handler.environment.update_state(initial_state)
        
        self.console.print(Panel(
            "Starting multi-agent collaboration",
            title="Orchestrator",
            border_style="bold blue"
        ))
        
        # Run with custom stop condition
        def stop_condition(state):
            # Stop when all queues are empty and no pending work
            no_files = not state.get("files_to_process", [])
            no_tasks = all(
                not tasks for tasks in state.get("task_queue", {}).values()
            )
            return no_files and no_tasks
        
        self.handler.run(max_steps=max_steps, stop_condition=stop_condition)