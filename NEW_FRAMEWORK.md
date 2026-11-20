# New Agent Framework Documentation

## Overview

The agent framework has been redesigned to follow a step-based execution model with shared environments. This enables better state management, multi-agent collaboration, and clearer separation of concerns.

## Core Concepts

### 1. Environment
- **Purpose**: Manages shared state between agents
- **Types**:
  - `SharedEnvironment`: All agents share the same state
  - `IsolatedEnvironment`: Each agent has its own state namespace with optional shared state
- **Location**: `core/environment.py`

### 2. Agent Base Class
- **Step Function**: Every agent must implement `step(environment)` 
- **State Flow**:
  1. Ingest state from environment
  2. Make decisions using LM
  3. Execute tools against environment
- **Location**: `core/base.py`

### 3. Step-Based Execution
- Agents execute one step at a time
- Handler orchestrates multiple agents
- Clear state transitions and visibility

## Key Improvements

### 1. Environment Management
```python
# Agents can share an environment
env = SharedEnvironment()
agent1 = MyAgent(lm, env)
agent2 = MyAgent(lm, env)

# Or have isolated environments
env = IsolatedEnvironment()
```

### 2. Step Function Pattern
```python
def step(self, environment: Environment) -> Dict[str, Any]:
    # 1. Read state
    state = self.ingest_state(environment)
    
    # 2. Process
    result = self.process_task(state)
    
    # 3. Update environment
    environment.update_state({"result": result}, agent=self)
    
    return {"status": "completed", "result": result}
```

### 3. Invoke Method Enhancement
The `invoke()` method now:
- Automatically injects environment state context
- Maintains conversation history
- Returns token usage information

## Usage Examples

### Single Agent
```python
env = SharedEnvironment()
agent = FileProcessorAgent(lm, env)
handler = StepHandler(env)
handler.add_agent(agent)
handler.run(max_steps=10)
```

### Multi-Agent Collaboration
```python
orchestrator = MultiAgentOrchestrator()
orchestrator.add_agent(coordinator, role="Coordinator")
orchestrator.add_agent(analyzer, role="Analyzer")
orchestrator.run_collaboration(initial_state, max_steps=20)
```

## File Structure
```
core/
  - base.py          # Updated Agent base class with step()
  - environment.py   # New Environment classes
  
agents/
  - example_agents.py    # FileProcessorAgent, CollaborativeAgent
  - code_editor.py       # Updated with step() implementation
  
handler/
  - step_handler.py      # New step-based execution handler
  
demo_new_framework.py   # Comprehensive demos
```

## Migration Guide

### Old Pattern
```python
class MyAgent(Agent):
    def __init__(self, lm):
        super().__init__(lm)
        self._environment["data"] = initial_data
```

### New Pattern
```python
class MyAgent(Agent):
    def __init__(self, lm, environment=None):
        super().__init__(lm, environment)
        self.environment.update_state({"data": initial_data}, agent=self)
    
    def step(self, environment):
        state = self.ingest_state(environment)
        # Process step
        return {"status": "completed"}
```

## Benefits

1. **Clearer State Management**: Environment explicitly manages state
2. **Better Multi-Agent Support**: Shared environments enable collaboration
3. **Step-Based Clarity**: Each step is atomic and trackable
4. **Flexible Execution**: Handlers can implement different execution strategies
5. **Token Tracking**: Built-in support for monitoring LLM usage

## Running Demos

```bash
# Run all demos
python demo_new_framework.py

# Run specific demo
python demo_new_framework.py 1  # Single agent
python demo_new_framework.py 2  # Multi-agent
python demo_new_framework.py 3  # Isolated environments
python demo_new_framework.py 4  # Custom step logic
```