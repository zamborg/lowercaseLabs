import pytest
from zagency.core.base import tool


def test_tool_decorator():
    """Test that the tool decorator adds the _is_tool attribute."""
    
    @tool
    def sample_function():
        """A sample function."""
        return "test"
    
    assert hasattr(sample_function, "_is_tool")
    assert sample_function._is_tool is True
    assert sample_function() == "test"
    assert sample_function.__doc__ == "A sample function."


def test_tool_decorator_with_arguments():
    """Test that the tool decorator works with functions that have arguments."""
    
    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    assert hasattr(add_numbers, "_is_tool")
    assert add_numbers._is_tool is True
    assert add_numbers(2, 3) == 5


def test_tool_decorator_preserves_function_metadata():
    """Test that the tool decorator preserves function metadata."""
    
    @tool
    def complex_function(x, y=10, *args, **kwargs):
        """A complex function with various parameter types."""
        return x + y + sum(args) + sum(kwargs.values())
    
    assert hasattr(complex_function, "_is_tool")
    assert complex_function._is_tool is True
    assert complex_function.__name__ == "complex_function"
    assert "complex function" in complex_function.__doc__
    
    # Test the function still works normally
    assert complex_function(1, 2, 3, 4, extra=5) == 15  # 1+2+3+4+5 