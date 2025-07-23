"""
This module contains the core base classes for the agentic framework.
"""

import inspect
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

import litellm
from pydantic import create_model
from .environment import Environment, SharedEnvironment

def tool(func: Callable) -> Callable:
    """
    Decorator to mark a method as a tool that can be called by an LM.
    """
    func._is_tool = True
    return func

class LM(ABC):
    """
    Abstract Base class for all LMs.
    """

    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model

    @abstractmethod
    def invoke(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
        """
        Invokes the LM.
        """
        pass

class LiteLLM(LM):
    """
    A concrete implementation of an LM that uses litellm.
    """

    def invoke(
        self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> dict:
        """
        Invokes the LM with function calling and returns both the response and token usage.
        """
        response = litellm.completion(
            model=self.model,
            messages=messages,
            tools=[{"type": "function", "function": t} for t in tools],
        )
        # Extract token usage if available
        usage = getattr(response, 'usage', None)
        if usage is None and hasattr(response, '__getitem__'):
            usage = response.get('usage', None)
        return {"response": response, "usage": usage}

class Agent(ABC):
    """
    Base class for all agents. An Agent has an environment (its state),
    an LM (its brain), and a set of tools (its methods).
    
    The agent follows a step-based execution model:
    1. Ingest state from environment
    2. Call LM with current state and history
    3. Execute tools against the environment
    """

    def __init__(self, lm: LM, environment: Optional[Environment] = None):
        self.environment = environment or SharedEnvironment()
        self.environment.register_agent(self)
        self.lm = lm
        self.tools = self._discover_tools()
        self.is_finished = False
        self._internal_state = {}  # Agent's private state
        self._history = []  # Conversation history
    
    @abstractmethod
    def step(self, environment: Environment) -> Dict[str, Any]:
        """
        Execute one step of the agent's logic.
        This is where the agent:
        1. Reads state from the environment
        2. Decides what to do
        3. Updates its internal state
        4. Returns the result of the step
        """
        pass
    
    def ingest_state(self, environment: Environment) -> Dict[str, Any]:
        """
        Read and process state from the environment.
        Subclasses can override to customize state ingestion.
        """
        return environment.get_state(self)

    def _discover_tools(self) -> Dict[str, Callable]:
        """Finds all methods decorated with @tool."""
        tools = {}
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if hasattr(method, "_is_tool"):
                tools[name] = method
        return tools

    def _generate_tool_definitions(self) -> List[Dict[str, Any]]:
        """Generates OpenAI-compatible tool definitions from the agent's methods."""
        definitions = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            doc = inspect.getdoc(func)

            fields = {
                param.name: (param.annotation, ...)
                for param in sig.parameters.values()
                if param.name != "self"
            }
            params_model = create_model(f"{name}Params", **fields)
            schema = params_model.model_json_schema()

            definitions.append(
                {
                    "name": name,
                    "description": doc,
                    "parameters": {
                        "type": "object",
                        "properties": schema.get("properties", {}),
                        "required": schema.get("required", []),
                    },
                }
            )
        return definitions

    def _execute_tool_call(self, tool_call) -> Any:
        """Executes a tool call and returns the result."""
        func_name = tool_call.function.name
        if func_name in self.tools:
            kwargs = json.loads(tool_call.function.arguments)
            return self.tools[func_name](**kwargs)
        else:
            return f"Error: Tool '{func_name}' not found."

    def invoke(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        The main entry point for the agent following the pattern:
        1. Ingest state from environment
        2. Call LM with state context
        3. Execute tools against environment
        """
        # Step 1: Ingest state from environment
        current_state = self.ingest_state(self.environment)
        
        # Add state context to messages if needed
        state_context = self._format_state_for_lm(current_state)
        if state_context:
            # Inject state context as a system message
            messages_with_state = [
                {"role": "system", "content": f"Current environment state: {state_context}"}
            ] + messages
        else:
            messages_with_state = messages
        
        # Step 2: Call LM with tools
        tool_defs = self._generate_tool_definitions()
        lm_result = self.lm.invoke(messages_with_state, tool_defs)
        
        # Handle response
        if isinstance(lm_result, dict) and "response" in lm_result:
            response = lm_result["response"]
            usage = lm_result.get("usage", None)
        else:
            response = lm_result
            usage = None
        response_message = response.choices[0].message

        result = {"assistant_message": response_message}
        if usage is not None:
            result["token_usage"] = usage

        # Step 3: Execute tools against environment
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            observation = self._execute_tool_call(tool_call)
            result["observation_message"] = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": str(observation),
            }
        
        # Update history
        self._history.append(messages[-1])  # User message
        self._history.append(response_message)  # Assistant response
        
        return result
    
    def _format_state_for_lm(self, state: Dict[str, Any]) -> str:
        """
        Format environment state for LM context.
        Override in subclasses for custom formatting.
        """
        if not state:
            return ""
        return str(state)

    @tool
    def exit(self):
        """
        Stops the agent's execution loop. Call this when the task is complete.
        """
        self.is_finished = True
        return "Agent execution has been stopped."
