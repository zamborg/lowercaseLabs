# QirkTheCoder Testing Framework

This directory contains the testing and evaluation framework for QirkTheCoder, our state-of-the-art coding agent.

## Overview

The testing framework uses YAML-based scenario definitions to create reproducible test environments. Each scenario defines:
- Initial file state
- Tasks for the agent to complete
- Expected outcomes
- Evaluation criteria

## Directory Structure

```
qirk_tests/
├── scenarios/          # YAML test scenario definitions
├── fixtures/           # Temporary test projects created during tests
├── results/           # Test run results (JSON format)
├── environment_loader.py   # Loads scenarios from YAML
├── test_runner.py     # Main test execution script
├── test_qirk_tools.py # Unit tests for individual tools
└── environment_schema.yaml # Documentation of YAML schema
```

## Running Tests

### Run all scenarios:
```bash
python qirk_tests/test_runner.py
```

### Run a specific scenario:
```bash
python qirk_tests/test_runner.py qirk_tests/scenarios/simple_test_completion.yaml
```

### Run with options:
```bash
python qirk_tests/test_runner.py \
    --model gpt-4-turbo \
    --max-steps 30 \
    --output results/test_run.json \
    --verbose
```

### Run unit tests:
```bash
pytest qirk_tests/test_qirk_tools.py -v
```

## Available Test Scenarios

1. **simple_test_completion.yaml**
   - Complete unit tests for a greeting function
   - Tests basic code understanding and test writing

2. **bug_fix_zero_division.yaml**
   - Fix zero division errors in calculator code
   - Tests debugging and error handling

3. **refactor_to_class.yaml**
   - Refactor procedural code into OOP design
   - Tests code structure understanding and refactoring

4. **implement_feature.yaml**
   - Add search functionality to a note-taking app
   - Tests feature implementation and API design

## Creating New Scenarios

1. Create a new YAML file in `scenarios/`
2. Follow the schema defined in `environment_schema.yaml`
3. Define:
   - Initial files and state
   - Tasks (or let agent infer from context)
   - Expected outcomes
   - Evaluation criteria

Example minimal scenario:
```yaml
environment:
  metadata:
    name: "My Test"
    description: "Test description"
    difficulty: "easy"
    tags: ["feature"]
    
  initial_state:
    files:
      - path: "main.py"
        content: |
          # Code here
    
  expectations:
    tests_should_pass: true
    files_should_contain:
      - path: "main.py"
        contains: ["expected_function"]
```

## Evaluation System

The framework evaluates agent performance based on:

### Success Criteria
- **all_files_exist**: Required files are present
- **content_checks_pass**: Files contain expected content
- **tests_pass**: Test suite passes
- **no_forbidden_content**: No TODO/FIXME markers remain
- **custom_checks_pass**: Custom Python expressions evaluate to True

### Scoring
Each criterion can have a point value. Total score helps compare performance across scenarios.

### Metrics Tracked
- Total steps taken
- Execution time
- Files modified
- Test results
- Errors encountered

## Environment Loading

The `EnvironmentLoader` class:
1. Creates a temporary project directory
2. Populates it with initial files
3. Sets up git repository (if specified)
4. Initializes environment state
5. Tracks all modifications

The agent can read the environment state to understand:
- What files exist
- What tasks are pending
- Test results
- Previous modifications

## Integration with Agent Framework

QirkTheCoder integrates with the testing framework through:
1. Reading tasks from `pending_tasks` in environment state
2. Using tools to modify files and run tests
3. Updating environment state with results
4. Step-based execution tracked by the handler

The agent can figure out what to do by:
- Reading explicit tasks from state
- Inferring from test failures
- Understanding from file TODOs
- Analyzing scenario metadata

## Best Practices

1. **Start Simple**: Begin with easy scenarios to verify setup
2. **Clear Expectations**: Define precise success criteria
3. **Realistic Tasks**: Create scenarios that mirror real coding tasks
4. **Incremental Difficulty**: Progress from simple to complex
5. **Comprehensive Checks**: Test both positive and negative cases

## Future Enhancements

- Parallel test execution
- Performance benchmarking
- Comparison across different models
- Integration with CI/CD
- Visual test result dashboard
- Automatic scenario generation from real issues