#!/usr/bin/env python3
"""
Demo script showcasing the new agent framework with step-based execution
and shared environments.
"""

from core.base import LiteLLM
from core.environment import SharedEnvironment, IsolatedEnvironment
from agents.example_agents import FileProcessorAgent, CollaborativeAgent
from handler.step_handler import StepHandler, MultiAgentOrchestrator


def demo_single_agent():
    """Demo of a single agent processing files."""
    print("\n=== Single Agent Demo: File Processor ===\n")
    
    # Create environment with files to process
    env = SharedEnvironment()
    env.update_state({
        "files_to_process": [
            {"name": "report.txt", "content": "This is a quarterly report with important metrics."},
            {"name": "memo.txt", "content": "Team meeting scheduled for next Monday."},
            {"name": "data.csv", "content": "id,name,value\\n1,item1,100\\n2,item2,200"}
        ],
        "processed_files": []
    })
    
    # Create agent
    lm = LiteLLM(model="gpt-4-turbo")
    agent = FileProcessorAgent(lm, env)
    
    # Create handler and run
    handler = StepHandler(env)
    handler.add_agent(agent)
    
    # Run until all files are processed
    handler.run(max_steps=10)


def demo_multi_agent_collaboration():
    """Demo of multiple agents collaborating through shared environment."""
    print("\n=== Multi-Agent Collaboration Demo ===\n")
    
    # Create shared environment
    env = SharedEnvironment()
    
    # Create collaborative agents
    lm = LiteLLM(model="gpt-4-turbo")
    
    coordinator = CollaborativeAgent(lm, env, "Coordinator")
    analyzer = CollaborativeAgent(lm, env, "Analyzer")
    reporter = CollaborativeAgent(lm, env, "Reporter")
    
    # Set up initial state with tasks
    initial_state = {
        "task_queue": {
            "Coordinator": [
                {"description": "Create analysis tasks for the data files"},
                {"description": "Coordinate report generation"}
            ],
            "Analyzer": [],
            "Reporter": []
        },
        "data_files": [
            {"name": "sales.csv", "data": [100, 200, 150, 300]},
            {"name": "costs.csv", "data": [50, 80, 70, 120]}
        ],
        "completed_tasks": [],
        "broadcast_messages": []
    }
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(env)
    orchestrator.add_agent(coordinator, role="Task Coordinator")
    orchestrator.add_agent(analyzer, role="Data Analyzer")
    orchestrator.add_agent(reporter, role="Report Generator")
    
    # Run collaboration
    orchestrator.run_collaboration(initial_state, max_steps=20)


def demo_isolated_environments():
    """Demo showing agents with isolated state namespaces."""
    print("\n=== Isolated Environments Demo ===\n")
    
    # Create isolated environment
    env = IsolatedEnvironment()
    
    # Create two file processor agents
    lm = LiteLLM(model="gpt-4-turbo")
    
    agent1 = FileProcessorAgent(lm, env)
    agent2 = FileProcessorAgent(lm, env)
    
    # Give each agent different files
    env.update_state({
        "files_to_process": [
            {"name": "agent1_file.txt", "content": "Data for agent 1"}
        ]
    }, agent=agent1)
    
    env.update_state({
        "files_to_process": [
            {"name": "agent2_file.txt", "content": "Data for agent 2"}
        ]
    }, agent=agent2)
    
    # Create handler
    handler = StepHandler(env)
    handler.add_agent(agent1)
    handler.add_agent(agent2)
    
    # Run - each agent processes its own files
    handler.run(max_steps=5)


def demo_custom_step_logic():
    """Demo of an agent with custom step logic."""
    print("\n=== Custom Step Logic Demo ===\n")
    
    from core.base import Agent, tool
    
    class CounterAgent(Agent):
        """Agent that counts up to a target value."""
        
        def __init__(self, lm, environment, name):
            super().__init__(lm, environment)
            self.name = name
            self.counter = 0
        
        def step(self, environment):
            state = self.ingest_state(environment)
            target = state.get(f"{self.name}_target", 5)
            
            if self.counter >= target:
                return {"status": "finished", "counter": self.counter}
            
            # Increment counter
            self.counter += 1
            
            # Use LM to generate a message about the count
            messages = [{
                "role": "user",
                "content": f"Generate a creative message about reaching count {self.counter}"
            }]
            
            result = self.invoke(messages)
            
            # Update shared state
            counts = state.get("all_counts", {})
            counts[self.name] = self.counter
            environment.update_state({"all_counts": counts}, agent=self)
            
            return {
                "status": "counting",
                "counter": self.counter,
                "target": target,
                "message": result.get("assistant_message", {}).get("content", "")
            }
        
        @tool
        def announce_milestone(self, milestone: int) -> str:
            """Announce reaching a milestone number."""
            if self.counter == milestone:
                return f"{self.name} reached milestone {milestone}!"
            return f"{self.name} is at {self.counter}, not at milestone {milestone}"
    
    # Create environment with different targets
    env = SharedEnvironment()
    env.update_state({
        "AgentA_target": 3,
        "AgentB_target": 5,
        "all_counts": {}
    })
    
    # Create agents
    lm = LiteLLM(model="gpt-4-turbo")
    agent_a = CounterAgent(lm, env, "AgentA")
    agent_b = CounterAgent(lm, env, "AgentB")
    
    # Run
    handler = StepHandler(env)
    handler.add_agent(agent_a)
    handler.add_agent(agent_b)
    handler.run(max_steps=10)


if __name__ == "__main__":
    import sys
    
    demos = {
        "1": ("Single Agent File Processing", demo_single_agent),
        "2": ("Multi-Agent Collaboration", demo_multi_agent_collaboration),
        "3": ("Isolated Environments", demo_isolated_environments),
        "4": ("Custom Step Logic", demo_custom_step_logic)
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in demos:
        name, func = demos[sys.argv[1]]
        print(f"Running: {name}")
        func()
    else:
        print("Available demos:")
        for key, (name, _) in demos.items():
            print(f"  {key}: {name}")
        print("\nUsage: python demo_new_framework.py <demo_number>")
        print("\nRunning all demos...\n")
        
        for key, (name, func) in demos.items():
            func()
            print("\n" + "="*60 + "\n")