"""
This module contains the core base classes for the agentic framework.
"""

import inspect
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List

import litellm
from pydantic import create_model

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
    """

    def __init__(self, lm: LM):
        self._environment = {}
        self.lm = lm
        self.tools = self._discover_tools()
        self.is_finished = False

    @property
    def environment(self) -> Dict[str, Any]:
        """
        The environment of the agent.
        """
        return self._environment

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
        The main entry point for the agent.
        It uses its LM to decide which tool to call based on the provided
        message history, then executes it.
        Now also returns token usage if available.
        """
        tool_defs = self._generate_tool_definitions()
        lm_result = self.lm.invoke(messages, tool_defs)
        # Support both old and new LM return types
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

        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            observation = self._execute_tool_call(tool_call)
            result["observation_message"] = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": str(observation),
            }
        return result

    @tool
    def exit(self):
        """
        Stops the agent's execution loop. Call this when the task is complete.
        """
        self.is_finished = True
        return "Agent execution has been stopped."
