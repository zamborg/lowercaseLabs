# Tests for Zagency

This directory contains comprehensive tests for all modules in the `zagency` package.

## Running Tests

To run all tests:
```bash
make test
```

Or run pytest directly:
```bash
python -m pytest tests/ -v
```

## Test Structure

The test directory mirrors the structure of the `zagency` package:

```
tests/
├── conftest.py           # Shared fixtures and test configuration
├── core/                 # Tests for zagency.core modules
│   ├── test_agent.py     # Agent class tests
│   ├── test_base.py      # Tool decorator tests  
│   ├── test_environment.py  # Environment classes tests
│   ├── test_lm.py        # Language model classes tests
│   └── test_template_loader.py  # Template loader tests
├── handler/              # Tests for zagency.handler modules
│   └── test_step_handler.py     # Step handler and orchestrator tests
└── environments/        # Tests for zagency.environments modules
    └── test_coding_environment.py  # Coding environment tests
```

## What's Tested

### Core Modules
- **Agent**: Abstract agent functionality, tool discovery, LM integration, tool execution
- **Base**: Tool decorator functionality
- **Environment**: Shared environment state management and agent registration
- **LM**: Language model abstraction and LiteLLM implementation
- **Template Loader**: YAML configuration loading with environment variable overrides

### Handler Modules  
- **Step Handler**: Multi-agent step-based execution, orchestration, and error handling

### Environment Modules
- **Coding Environment**: File state management, patch application, test running, project structure analysis

## Test Coverage

The test suite includes:
- **92 tests** covering all major functionality
- **Unit tests** with comprehensive mocking
- **Integration tests** for multi-component interactions
- **Error handling** and edge case testing
- **Abstract class** enforcement testing
- **File system operations** with temporary directories
- **Configuration loading** from multiple sources

## Dependencies

Test dependencies are specified in `tests/requirements.txt`:
- `pytest>=7.0.0` - Testing framework
- `pytest-mock>=3.6.0` - Enhanced mocking capabilities

## Adding New Tests

When adding new functionality to zagency:

1. Create corresponding test files in the appropriate test subdirectory
2. Use the shared fixtures from `conftest.py` for common test setup
3. Follow the existing naming convention: `test_<module_name>.py`
4. Group related tests into test classes
5. Use descriptive test method names that explain what is being tested

## Running Specific Tests

Run tests for a specific module:
```bash
python -m pytest tests/core/test_agent.py -v
```

Run a specific test:
```bash
python -m pytest tests/core/test_agent.py::TestAgent::test_agent_init -v
```

Run tests with coverage:
```bash
python -m pytest tests/ --cov=zagency --cov-report=html
``` 