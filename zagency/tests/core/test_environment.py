import pytest
from zagency.core.environment import Environment, SharedEnvironment


class MockAgent:
    """Mock agent for testing environment registration."""
    
    def __init__(self, name):
        self.name = name
    
    def __eq__(self, other):
        return isinstance(other, MockAgent) and self.name == other.name


class TestEnvironment:
    """Test the abstract Environment base class."""
    
    def test_environment_init(self):
        """Test Environment initialization."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            Environment()


class TestSharedEnvironment:
    """Test the SharedEnvironment implementation."""
    
    def test_shared_environment_init(self):
        """Test SharedEnvironment initialization."""
        env = SharedEnvironment()
        assert env._state == {}
        assert env._agents == []
    
    def test_register_agent(self):
        """Test registering an agent."""
        env = SharedEnvironment()
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")
        
        env.register_agent(agent1)
        assert len(env._agents) == 1
        assert agent1 in env._agents
        
        env.register_agent(agent2)
        assert len(env._agents) == 2
        assert agent2 in env._agents
        
        # Test registering same agent twice
        env.register_agent(agent1)
        assert len(env._agents) == 2  # Should not duplicate
    
    def test_unregister_agent(self):
        """Test unregistering an agent."""
        env = SharedEnvironment()
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")
        
        env.register_agent(agent1)
        env.register_agent(agent2)
        assert len(env._agents) == 2
        
        env.unregister_agent(agent1)
        assert len(env._agents) == 1
        assert agent1 not in env._agents
        assert agent2 in env._agents
        
        # Test unregistering agent not in list
        env.unregister_agent(agent1)
        assert len(env._agents) == 1  # Should not error
    
    def test_get_state(self):
        """Test getting environment state."""
        env = SharedEnvironment()
        env._state = {"key1": "value1", "key2": 42}
        
        state = env.get_state()
        assert state == {"key1": "value1", "key2": 42}
        
        # Test that it returns a copy
        state["key3"] = "value3"
        assert "key3" not in env._state
    
    def test_get_state_with_agent(self):
        """Test getting state with specific agent (should be same for SharedEnvironment)."""
        env = SharedEnvironment()
        env._state = {"shared": "data"}
        agent = MockAgent("test")
        
        state = env.get_state(agent)
        assert state == {"shared": "data"}
    
    def test_update_state(self):
        """Test updating environment state."""
        env = SharedEnvironment()
        
        updates = {"key1": "value1", "key2": 42}
        env.update_state(updates)
        assert env._state == updates
        
        # Test partial updates
        more_updates = {"key3": "value3", "key1": "new_value1"}
        env.update_state(more_updates)
        assert env._state == {"key1": "new_value1", "key2": 42, "key3": "value3"}
    
    def test_update_state_with_agent(self):
        """Test updating state with specific agent (should work same for SharedEnvironment)."""
        env = SharedEnvironment()
        agent = MockAgent("test")
        
        updates = {"agent_data": "test"}
        env.update_state(updates, agent)
        assert env._state == updates
    
    def test_dict_like_access(self):
        """Test dict-like access to environment state."""
        env = SharedEnvironment()
        
        # Test __setitem__
        env["key1"] = "value1"
        env["key2"] = 42
        assert env._state == {"key1": "value1", "key2": 42}
        
        # Test __getitem__
        assert env["key1"] == "value1"
        assert env["key2"] == 42
        assert env["nonexistent"] is None
    
    def test_dict_access_with_none_values(self):
        """Test dict access with None values."""
        env = SharedEnvironment()
        env["key"] = None
        
        assert env["key"] is None
        assert env["nonexistent"] is None 