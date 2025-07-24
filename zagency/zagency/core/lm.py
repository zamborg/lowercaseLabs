from abc import ABC, abstractmethod
from typing import Any, Dict, List

import litellm


class LM(ABC):
    """
    Abstract Base class for all LMs.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def invoke(self, **kwargs) -> Any:
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