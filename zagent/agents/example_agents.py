"""
Example agents demonstrating the new step-based framework.
"""

from typing import Any, Dict
from core.base import Agent, LM, tool
from core.environment import Environment, SharedEnvironment


class FileProcessorAgent(Agent):
    """
    An agent that processes files in a shared environment.
    Demonstrates the step function and environment interaction pattern.
    """
    
    def __init__(self, lm: LM, environment: Environment):
        super().__init__(lm, environment)
        self._current_file = None
    
    def step(self, environment: Environment) -> Dict[str, Any]:
        """
        Execute one step: check for files to process, process them, update state.
        """
        # 1. Ingest state from environment
        state = self.ingest_state(environment)
        
        # Check if there are files to process
        files_to_process = state.get("files_to_process", [])
        
        if not files_to_process:
            return {"status": "idle", "message": "No files to process"}
        
        # Get the next file
        current_file = files_to_process[0]
        self._current_file = current_file
        
        # 2. Use invoke to let LM decide what to do with the file
        messages = [{
            "role": "user",
            "content": f"Process the file: {current_file['name']} with content: {current_file['content']}"
        }]
        
        result = self.invoke(messages)
        
        # 3. Update environment state
        remaining_files = files_to_process[1:]
        processed_files = state.get("processed_files", [])
        processed_files.append({
            "file": current_file,
            "result": result.get("assistant_message", {}).get("content", ""),
            "timestamp": state.get("current_time", "unknown")
        })
        
        environment.update_state({
            "files_to_process": remaining_files,
            "processed_files": processed_files
        }, agent=self)
        
        return {
            "status": "processed",
            "file": current_file["name"],
            "remaining": len(remaining_files)
        }
    
    @tool
    def analyze_content(self, analysis_type: str) -> str:
        """
        Analyze the content of the current file.
        
        Args:
            analysis_type: Type of analysis to perform (e.g., 'sentiment', 'summary', 'keywords')
        """
        if not self._current_file:
            return "No file is currently being processed"
        
        content = self._current_file.get("content", "")
        
        # Simulate different types of analysis
        if analysis_type == "summary":
            return f"Summary of {self._current_file['name']}: {content[:50]}..."
        elif analysis_type == "keywords":
            words = content.split()[:5]
            return f"Keywords: {', '.join(words)}"
        elif analysis_type == "sentiment":
            return "Sentiment: Neutral"
        else:
            return f"Unknown analysis type: {analysis_type}"
    
    @tool
    def transform_content(self, transformation: str) -> str:
        """
        Transform the content of the current file.
        
        Args:
            transformation: Type of transformation ('uppercase', 'lowercase', 'reverse')
        """
        if not self._current_file:
            return "No file is currently being processed"
        
        content = self._current_file.get("content", "")
        
        if transformation == "uppercase":
            transformed = content.upper()
        elif transformation == "lowercase":
            transformed = content.lower()
        elif transformation == "reverse":
            transformed = content[::-1]
        else:
            return f"Unknown transformation: {transformation}"
        
        # Update the file content in internal state
        self._current_file["transformed_content"] = transformed
        return f"Content transformed using {transformation}"


class CollaborativeAgent(Agent):
    """
    An agent that collaborates with other agents through shared environment.
    """
    
    def __init__(self, lm: LM, environment: Environment, agent_name: str):
        super().__init__(lm, environment)
        self.agent_name = agent_name
    
    def step(self, environment: Environment) -> Dict[str, Any]:
        """
        Check for tasks assigned to this agent and execute them.
        """
        state = self.ingest_state(environment)
        
        # Check for tasks assigned to this agent
        task_queue = state.get("task_queue", {})
        my_tasks = task_queue.get(self.agent_name, [])
        
        if not my_tasks:
            return {"status": "waiting", "agent": self.agent_name}
        
        # Get the next task
        current_task = my_tasks[0]
        
        # Process the task using LM
        messages = [{
            "role": "user",
            "content": f"Execute task: {current_task['description']}"
        }]
        
        result = self.invoke(messages)
        
        # Update task queue
        remaining_tasks = my_tasks[1:]
        task_queue[self.agent_name] = remaining_tasks
        
        # Record completed task
        completed_tasks = state.get("completed_tasks", [])
        completed_tasks.append({
            "agent": self.agent_name,
            "task": current_task,
            "result": result.get("assistant_message", {}).get("content", "")
        })
        
        environment.update_state({
            "task_queue": task_queue,
            "completed_tasks": completed_tasks
        }, agent=self)
        
        return {
            "status": "completed",
            "agent": self.agent_name,
            "task": current_task["description"]
        }
    
    @tool
    def create_task(self, target_agent: str, task_description: str) -> str:
        """
        Create a task for another agent.
        
        Args:
            target_agent: Name of the agent to assign the task to
            task_description: Description of the task
        """
        state = self.environment.get_state(self)
        task_queue = state.get("task_queue", {})
        
        if target_agent not in task_queue:
            task_queue[target_agent] = []
        
        task_queue[target_agent].append({
            "description": task_description,
            "created_by": self.agent_name
        })
        
        self.environment.update_state({"task_queue": task_queue}, agent=self)
        
        return f"Task created for {target_agent}: {task_description}"
    
    @tool
    def broadcast_message(self, message: str) -> str:
        """
        Broadcast a message to all agents via the shared environment.
        
        Args:
            message: The message to broadcast
        """
        state = self.environment.get_state(self)
        messages = state.get("broadcast_messages", [])
        
        messages.append({
            "from": self.agent_name,
            "message": message
        })
        
        self.environment.update_state({"broadcast_messages": messages}, agent=self)
        
        return f"Message broadcasted: {message}"