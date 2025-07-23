# zagent

An agentic framework for building AI agents with LLM integration.

## Features

- **Abstract base classes** for creating AI agents
- **LLM integration** with function calling support  
- **Tool discovery and execution** - automatically discover methods decorated with `@tool`
- **Multi-provider support** - built on top of LiteLLM for compatibility with OpenAI, Anthropic, and more
- **Token usage tracking** - monitor API usage and costs
- **Rich integration** - beautiful console output support

## Installation

```bash
pip install zagent
```

## Quick Start

```python
from zagent import Agent, LiteLLM, tool

class MyAgent(Agent):
    @tool
    def greet(self, name: str) -> str:
        """Greet someone by name"""
        return f"Hello, {name}!"
    
    @tool
    def calculate(self, a: int, b: int, operation: str) -> int:
        """Perform basic math operations"""
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        else:
            return 0

# Create an agent with an LLM
lm = LiteLLM(model="gpt-4")
agent = MyAgent(lm)

# Use the agent
messages = [{"role": "user", "content": "Please greet Alice and then calculate 5 + 3"}]
result = agent.invoke(messages)
print(result)
```

## Architecture

The framework consists of:

- **`Agent`** - Base class for all agents with tool discovery and execution
- **`LM`** - Abstract base class for language models
- **`LiteLLM`** - Concrete LM implementation using LiteLLM
- **`@tool`** - Decorator to mark methods as agent tools
- **`Handler`** - Request handling utilities

## Development

### Building and Publishing

This package uses a Makefile for easy development workflow:

```bash
# Show available commands
make help

# Clean up build artifacts
make cleanup

# Build the package
make build

# Check the built package
make check

# Full release process (build, check, and publish to PyPI)
make release
```

### Requirements

The package depends on:
- `torch` - For ML model support
- `whisper` - For audio processing
- `pyannote.audio` - For audio analysis
- `ffmpeg-python` - For media processing
- `litellm` - For LLM provider abstraction
- `rich` - For beautiful console output
- `pydantic` - For data validation

## License

MIT License