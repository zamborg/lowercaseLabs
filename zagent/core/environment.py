"""
Environment base class for shared state management across agents.
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


class IsolatedEnvironment(Environment):
    """
    An environment where each agent has its own isolated state namespace.
    """
    
    def __init__(self):
        super().__init__()
        self._agent_states: Dict[str, Dict[str, Any]] = {}
        self._shared_state: Dict[str, Any] = {}
    
    def register_agent(self, agent: "Agent"):
        """Register an agent and create its state namespace."""
        super().register_agent(agent)
        agent_id = id(agent)
        if agent_id not in self._agent_states:
            self._agent_states[agent_id] = {}
    
    def get_state(self, agent: Optional["Agent"] = None) -> Dict[str, Any]:
        """Get combined shared state and agent-specific state."""
        state = self._shared_state.copy()
        if agent:
            agent_id = id(agent)
            if agent_id in self._agent_states:
                state.update(self._agent_states[agent_id])
        return state
    
    def update_state(self, updates: Dict[str, Any], agent: Optional["Agent"] = None):
        """Update agent-specific state or shared state."""
        if agent:
            agent_id = id(agent)
            if agent_id in self._agent_states:
                self._agent_states[agent_id].update(updates)
        else:
            self._shared_state.update(updates)