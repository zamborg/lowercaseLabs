"""
Environment base class for shared state management across agents.
TODO: @zamborg clean up
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Environment(ABC):
    """
    Base class for environments that agents can interact with.
    Environments hold state that can be shared across multiple agents.
    """
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._agents: List["Agent"] = []
    
    def register_agent(self, agent: "Agent"):
        """Register an agent with this environment."""
        if agent not in self._agents:
            self._agents.append(agent)
    
    def unregister_agent(self, agent: "Agent"):
        """Unregister an agent from this environment."""
        if agent in self._agents:
            self._agents.remove(agent)
    
    @abstractmethod
    def get_state(self, agent: Optional["Agent"] = None) -> Dict[str, Any]:
        """
        Get the current state of the environment.
        Can optionally filter state based on the requesting agent.
        """
        pass
    
    @abstractmethod
    def update_state(self, updates: Dict[str, Any], agent: Optional["Agent"] = None):
        """
        Update the environment state.
        Can optionally track which agent made the update.
        """
        pass
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to state."""
        return self._state.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dict-like setting of state."""
        self._state[key] = value


class SharedEnvironment(Environment):
    """
    A simple shared environment where all agents see and can modify the same state.
    """
    
    def get_state(self, agent: Optional["Agent"] = None) -> Dict[str, Any]:
        """Return a copy of the entire state."""
        return self._state.copy()
    
    def update_state(self, updates: Dict[str, Any], agent: Optional["Agent"] = None):
        """Update the state with the provided updates."""
        self._state.update(updates)
