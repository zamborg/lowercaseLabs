"""
zagent - An agentic framework for building AI agents with LLM integration
"""

__version__ = "0.1.0"

from .core.base import Agent, LM, LiteLLM, tool
from .handler.handler import Handler

__all__ = ["Agent", "LM", "LiteLLM", "tool", "Handler"]