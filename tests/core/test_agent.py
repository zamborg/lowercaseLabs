import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from zagency.core.agent import Agent
from zagency.core.base import tool
from zagency.core.environment import SharedEnvironment


class MockAgent(Agent):
    """Concrete implementation of Agent for testing."""
    
    def step(self, environment):
        return {"status": "completed", "message": "Step executed"}
    
    def ingest_state(self, environment):
        return environment.get_state()
    
    def synthesize_lm_input(self, **kwargs):
        env_state = kwargs.get("env_state", {})
        return [
            {"role": "system", "content": "You are a test agent"},
            {"role": "user", "content": f"Current state: {env_state}"}
        ]
    
    @tool
    def test_tool(self, message: str) -> str:
        """A test tool that returns a message."""
        return f"Tool executed with: {message}"
    
    @tool
    def another_tool(self, value: int) -> int:
        """Another test tool that doubles a value."""
        return value * 2


class TestAgent:
    """Test the Agent base class."""
    
    def test_agent_init(self, mock_lm):
        """Test Agent initialization."""
        env = SharedEnvironment()
        agent = MockAgent(mock_lm, env)
        
        assert agent.environment == env
        assert agent.lm == mock_lm
        assert agent.is_finished is False
        assert agent._internal_state == {}
        assert agent.history == []
        assert agent._trajectory == []
        assert env._agents == [agent]  # Agent should be registered
    
    def test_discover_tools(self, mock_lm):
        """Test that tools are discovered correctly."""
        env = SharedEnvironment()
        agent = MockAgent(mock_lm, env)
        
        assert "test_tool" in agent.tools
        assert "another_tool" in agent.tools
        assert "exit" in agent.tools
        assert len(agent.tools) == 3
        
        # Test that tools are callable
        assert callable(agent.tools["test_tool"])
        assert callable(agent.tools["another_tool"])
    
    @patch('zagency.core.agent.create_model')
    def test_generate_tool_definitions(self, mock_create_model, mock_lm):
        """Test generating OpenAI-compatible tool definitions."""
        # Mock the Pydantic model creation
        mock_model = Mock()
        mock_model.model_json_schema.return_value = {
            "properties": {"message": {"type": "string"}},
            "required": ["message"]
        }
        mock_create_model.return_value = mock_model
        
        env = SharedEnvironment()
        agent = MockAgent(mock_lm, env)
        
        definitions = agent._generate_tool_definitions()
        
        assert len(definitions) == 3  # test_tool, another_tool, exit
        
        # Check test_tool definition
        test_tool_def = next(d for d in definitions if d["name"] == "test_tool")
        assert test_tool_def["description"] == "A test tool that returns a message."
        assert test_tool_def["parameters"]["type"] == "object"
        assert "properties" in test_tool_def["parameters"]
        assert "required" in test_tool_def["parameters"]
    
    def test_execute_tool_call(self, mock_lm):
        """Test executing a single tool call."""
        env = SharedEnvironment()
        agent = MockAgent(mock_lm, env)
        
        # Mock tool call
        tool_call = Mock()
        tool_call.function.name = "test_tool"
        tool_call.function.arguments = '{"message": "hello"}'
        
        result = agent._execute_tool_call(tool_call)
        
        assert result == {"test_tool": "Tool executed with: hello"}
    
    def test_execute_tool_call_nonexistent(self, mock_lm):
        """Test executing a tool call for a nonexistent tool."""
        env = SharedEnvironment()
        agent = MockAgent(mock_lm, env)
        
        # Mock tool call with nonexistent tool
        tool_call = Mock()
        tool_call.function.name = "nonexistent_tool"
        tool_call.function.arguments = '{}'
        
        result = agent._execute_tool_call(tool_call)
        
        assert result == "Error: Tool 'nonexistent_tool' not found."
    
    def test_execute_tool_calls(self, mock_lm):
        """Test executing multiple tool calls."""
        env = SharedEnvironment()
        agent = MockAgent(mock_lm, env)
        
        # Mock multiple tool calls
        tool_call1 = Mock()
        tool_call1.function.name = "test_tool"
        tool_call1.function.arguments = '{"message": "first"}'
        
        tool_call2 = Mock()
        tool_call2.function.name = "another_tool"
        tool_call2.function.arguments = '{"value": 5}'
        
        results = agent._execute_tool_calls([tool_call1, tool_call2])
        
        expected = {
            "test_tool": "Tool executed with: first",
            "another_tool": 10
        }
        assert results == expected
    
    def test_handle_lm_response_with_usage(self):
        """Test handling LM response with usage information."""
        mock_response = Mock()
        mock_response.choices = [Mock(message="test message")]
        
        lm_result = {
            "response": mock_response,
            "usage": {"total_tokens": 100}
        }
        
        result = Agent.handle_lm_response(lm_result)
        
        assert result["LM_response"] == "test message"
        assert result["token_usage"]["total_tokens"] == 100
    
    def test_handle_lm_response_without_usage(self):
        """Test handling LM response without usage information."""
        mock_response = Mock()
        mock_response.choices = [Mock(message="test message")]
        
        result = Agent.handle_lm_response(mock_response)
        
        assert result["LM_response"] == "test message"
        assert "token_usage" not in result
    
    @patch('zagency.core.agent.create_model')
    def test_invoke_without_tool_calls(self, mock_create_model, mock_lm):
        """Test agent invoke without tool calls."""
        # Setup mocks
        mock_model = Mock()
        mock_model.model_json_schema.return_value = {"properties": {}, "required": []}
        mock_create_model.return_value = mock_model
        
        mock_response_message = Mock()
        mock_response_message.tool_calls = None
        mock_lm.invoke.return_value = {
            "response": Mock(choices=[Mock(message=mock_response_message)]),
            "usage": {"total_tokens": 50}
        }
        
        env = SharedEnvironment()
        env.update_state({"test": "data"})
        agent = MockAgent(mock_lm, env)
        
        result = agent.invoke()
        
        assert result["role"] == "assistant"
        assert "content" in result
        
        # Verify LM was called with correct parameters
        mock_lm.invoke.assert_called_once()
        args, kwargs = mock_lm.invoke.call_args
        messages, tool_defs = args
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert len(tool_defs) == 3  # test_tool, another_tool, exit
    
    @patch('zagency.core.agent.create_model')
    def test_invoke_with_tool_calls(self, mock_create_model, mock_lm):
        """Test agent invoke with tool calls."""
        # Setup mocks
        mock_model = Mock()
        mock_model.model_json_schema.return_value = {"properties": {}, "required": []}
        mock_create_model.return_value = mock_model
        
        # Mock tool call
        tool_call = Mock()
        tool_call.function.name = "test_tool"
        tool_call.function.arguments = '{"message": "hello"}'
        
        mock_response_message = Mock()
        mock_response_message.tool_calls = [tool_call]
        mock_lm.invoke.return_value = {
            "response": Mock(choices=[Mock(message=mock_response_message)]),
            "usage": {"total_tokens": 75}
        }
        
        env = SharedEnvironment()
        agent = MockAgent(mock_lm, env)
        
        result = agent.invoke()
        
        assert result["role"] == "assistant"
        assert "content" in result
        assert "test_tool" in result["content"]
    
    def test_exit_tool(self, mock_lm):
        """Test the built-in exit tool."""
        env = SharedEnvironment()
        agent = MockAgent(mock_lm, env)
        
        assert agent.is_finished is False
        
        result = agent.exit()
        
        assert agent.is_finished is True
        assert "stopped" in result.lower()


class TestAgentAbstractMethods:
    """Test that abstract methods must be implemented."""
    
    def test_agent_requires_step_implementation(self, mock_lm):
        """Test that Agent subclasses must implement step()."""
        class IncompleteAgent(Agent):
            def ingest_state(self, environment):
                return {}
            
            def synthesize_lm_input(self, **kwargs):
                return []
        
        env = SharedEnvironment()
        # Should not be able to instantiate without step() implementation
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAgent(mock_lm, env)
    
    def test_agent_requires_ingest_state_implementation(self, mock_lm):
        """Test that Agent subclasses must implement ingest_state()."""
        class IncompleteAgent(Agent):
            def step(self, environment):
                return {}
            
            def synthesize_lm_input(self, **kwargs):
                return []
        
        env = SharedEnvironment()
        # Should not be able to instantiate without ingest_state() implementation
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAgent(mock_lm, env)
    
    def test_agent_requires_synthesize_lm_input_implementation(self, mock_lm):
        """Test that Agent subclasses must implement synthesize_lm_input()."""
        class IncompleteAgent(Agent):
            def step(self, environment):
                return {}
            
            def ingest_state(self, environment):
                return {}
        
        env = SharedEnvironment()
        # Should not be able to instantiate without synthesize_lm_input() implementation
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteAgent(mock_lm, env) 