import pytest
from unittest.mock import Mock, patch, MagicMock
from zagency.core.lm import LM, LiteLLM


class TestLM:
    """Test the abstract LM base class."""
    
    def test_lm_init(self):
        """Test LM initialization."""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            LM("test-model")


class TestLiteLLM:
    """Test the LiteLLM implementation."""
    
    def test_litellm_init(self):
        """Test LiteLLM initialization."""
        lm = LiteLLM("gpt-3.5-turbo")
        assert lm.model == "gpt-3.5-turbo"
    
    @patch('zagency.core.lm.litellm.completion')
    def test_invoke_with_tools(self, mock_completion):
        """Test invoking LiteLLM with tools."""
        # Setup mock response
        mock_response = Mock()
        mock_response.usage = {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50}
        mock_completion.return_value = mock_response
        
        lm = LiteLLM("gpt-3.5-turbo")
        messages = [{"role": "user", "content": "Hello"}]
        tools = [{"name": "test_tool", "description": "A test tool"}]
        
        result = lm.invoke(messages, tools)
        
        # Verify litellm.completion was called correctly
        mock_completion.assert_called_once_with(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=[{"type": "function", "function": t} for t in tools]
        )
        
        # Verify result structure
        assert "response" in result
        assert "usage" in result
        assert result["response"] == mock_response
        assert result["usage"]["total_tokens"] == 150
    
    @patch('zagency.core.lm.litellm.completion')
    def test_invoke_without_usage(self, mock_completion):
        """Test invoking LiteLLM when response doesn't have usage info."""
        # Setup mock response without usage
        mock_response = Mock()
        del mock_response.usage  # Remove usage attribute
        mock_completion.return_value = mock_response
        
        lm = LiteLLM("gpt-3.5-turbo")
        messages = [{"role": "user", "content": "Hello"}]
        tools = []
        
        result = lm.invoke(messages, tools)
        
        assert "response" in result
        assert "usage" in result
        assert result["response"] == mock_response
        assert result["usage"] is None
    
    @patch('zagency.core.lm.litellm.completion')
    def test_invoke_with_dict_response(self, mock_completion):
        """Test invoking LiteLLM when response is a dict with usage."""
        # Setup mock response as dict
        mock_response = {
            "choices": [{"message": {"content": "Hello!"}}],
            "usage": {"total_tokens": 25}
        }
        mock_completion.return_value = mock_response
        
        lm = LiteLLM("gpt-3.5-turbo")
        messages = [{"role": "user", "content": "Hello"}]
        tools = []
        
        result = lm.invoke(messages, tools)
        
        assert "response" in result
        assert "usage" in result
        assert result["response"] == mock_response
        assert result["usage"]["total_tokens"] == 25
    
    @patch('zagency.core.lm.litellm.completion')
    def test_invoke_error_handling(self, mock_completion):
        """Test error handling in LiteLLM invoke."""
        mock_completion.side_effect = Exception("API Error")
        
        lm = LiteLLM("gpt-3.5-turbo")
        messages = [{"role": "user", "content": "Hello"}]
        tools = []
        
        with pytest.raises(Exception) as exc_info:
            lm.invoke(messages, tools)
        
        assert str(exc_info.value) == "API Error" 