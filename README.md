# zagency

An agentic framework for building AI agents with LLM integration.

## Features

- **Abstract base classes** for creating AI agents
- **LLM integration** with function calling support  
- **Tool discovery and execution** - automatically discover methods decorated with `@tool`
- **Environment system** - shared state management between agents
- **Evaluation framework** - comprehensive scoring and evaluation system
- **Multi-provider support** - built on top of LiteLLM for compatibility with OpenAI, Anthropic, and more
- **Token usage tracking** - monitor API usage and costs
- **Rich integration** - beautiful console output support

## Installation

```bash
pip install zagency
```

## Quick Start

```python
from zagency import Agent, LiteLLM, tool

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
- **`Environment`** - State management system for agent interactions
- **`LM`** - Abstract base class for language models
- **`LiteLLM`** - Concrete LM implementation using LiteLLM
- **`@tool`** - Decorator to mark methods as agent tools
- **`Handler`** - Request handling utilities
- **`Scorer`** - Base classes for evaluating agent/environment behavior
- **`Evaluation`** - Framework for running agents on datasets with scoring


## 🏗️ Core Architecture

### 1. Agents (`core/base.py`)

All agents inherit from the `Agent` base class and implement the **step-based execution model**:

```python
class MyAgent(Agent):
    def __init__(self, lm: LM, environment: Environment = None):
        super().__init__(lm, environment)
    
    def step(self, environment: Environment) -> Dict[str, Any]:
        """Execute one step of the agent's logic"""
        # 1. Read state from environment
        state = self.ingest_state(environment)
        
        # 2. Process with LM (if needed)
        messages = [{"role": "user", "content": "task description"}]
        result = self.invoke(messages)
        
        # 3. Update environment
        environment.update_state({"result": result}, agent=self)
        
        return {"status": "completed"}
    
    @tool
    def my_tool(self, param: str) -> str:
        """Tools are decorated methods the LM can call"""
        return f"Processed: {param}"
```

**Key Features:**
- **Step function**: Atomic execution units for clear state transitions
- **Tool system**: Decorated methods automatically become LM-callable tools
- **Environment integration**: Shared state management across agents
- **LM abstraction**: Support for different language models via `LiteLLM`

### 2. Environments (`core/environment.py`)

Environments manage shared state between agents:

```python
# Shared environment - all agents see same state
env = SharedEnvironment()

# Isolated environment - each agent has private namespace
env = IsolatedEnvironment()
```

**Environment Types:**
- **SharedEnvironment**: All agents share identical state
- **IsolatedEnvironment**: Each agent has private state + shared globals
- **CodingEnvironment**: Specialized for code editing with file tracking

### 3. Handlers (`handler/step_handler.py`)

Handlers orchestrate agent execution:

```python
handler = StepHandler(environment)
handler.add_agent(agent1)
handler.add_agent(agent2)
handler.run(max_steps=50)
```

### 4. Scorers (`core/scorer.py`)

Scorers evaluate agent and environment behavior:

```python
# Agent scorer - evaluates agent internals
class ThinkingTokenScorer(AgentScorer):
    def score(self, agent: Agent, step_result=None):
        thinking_tokens = agent._internal_state.get("thinking_tokens", 0)
        total_tokens = agent._internal_state.get("total_tokens", 1)
        efficiency = 1.0 - (thinking_tokens / total_tokens)
        return {"score": efficiency, "thinking_tokens": thinking_tokens}

# Environment scorer - evaluates environment state
class FileAccessScorer(EnvironmentScorer):
    def score(self, environment: Environment, agent=None):
        file_accesses = environment.get_state().get("file_accesses", [])
        unique_files = set(file_accesses)
        redundancy_score = len(unique_files) / len(file_accesses)
        return {"score": redundancy_score, "blast_radius": len(unique_files)}

# Trace scorer - evaluates complete interactions
class CompletionTimeScorer(TraceScorer):
    def score(self, agent: Agent, environment: Environment, trace=None):
        num_steps = len(agent.history)
        efficiency = 1.0 / (1.0 + num_steps / 10.0) if agent.is_finished else 0.0
        return {"score": efficiency, "completed": agent.is_finished}
```

### 5. Evaluations (`core/evaluation.py`)

Run agents on datasets with scoring:

```python
# Create evaluation
evaluation = Evaluation(
    name="code_generation_eval",
    scorers=[ThinkingTokenScorer(), FileAccessScorer(), CompletionTimeScorer()]
)

# Run on dataset
dataset = [
    {"task": "implement fibonacci", "test_file": "test_fib.py"},
    {"task": "fix bug in parser", "test_file": "test_parser.py"}
]

results = evaluation.run(
    agent=coding_agent,
    environment_class=CodingEnvironment,
    dataset=dataset,
    max_steps_per_datum=50
)

print(f"Completion rate: {results['aggregate_scores']['completion_rate']:.2%}")
print(f"Average efficiency: {results['aggregate_scores']['scorer_aggregates']['CompletionTimeScorer']['mean']:.3f}")
```

## 🔧 How to Add New Components

### Adding a New Agent

1. **Create the agent class:**

```python
# agents/my_new_agent.py
from core.base import Agent, LM, tool
from typing import Dict, Any

class MyNewAgent(Agent):
    def __init__(self, lm: LM, environment=None, custom_param=None):
        super().__init__(lm, environment)
        self.custom_param = custom_param
        
    def step(self, environment) -> Dict[str, Any]:
        """Your step logic here"""
        state = self.ingest_state(environment)
        
        # Your processing logic
        if state.get("needs_processing"):
            messages = [{"role": "user", "content": "Process this data"}]
            result = self.invoke(messages)
            
            environment.update_state({
                "processing_complete": True,
                "result": result
            }, agent=self)
            
            return {"status": "processed"}
        
        return {"status": "idle"}
    
    @tool
    def custom_tool(self, data: str) -> str:
        """Custom tool for this agent"""
        return f"Processed {data} with {self.custom_param}"
```

2. **Create a demo script:**

```python
# demo_my_agent.py
from core.base import LiteLLM
from core.environment import SharedEnvironment
from agents.my_new_agent import MyNewAgent
from handler.step_handler import StepHandler

def demo():
    env = SharedEnvironment()
    env.update_state({"needs_processing": True})
    
    lm = LiteLLM(model="gpt-4-turbo")
    agent = MyNewAgent(lm, env, custom_param="example")
    
    handler = StepHandler(env)
    handler.add_agent(agent)
    handler.run(max_steps=10)

if __name__ == "__main__":
    demo()
```

### Adding a New Environment

1. **Inherit from Environment base class:**

```python
# environments/my_environment.py
from core.environment import Environment
from typing import Dict, Any, Optional

class MyCustomEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.custom_data = {}
    
    def get_state(self, agent: Optional["Agent"] = None) -> Dict[str, Any]:
        """Return state for the requesting agent"""
        return {
            **self._state,
            "custom_data": self.custom_data,
            "agent_specific": self._get_agent_data(agent)
        }
    
    def update_state(self, updates: Dict[str, Any], agent: Optional["Agent"] = None):
        """Handle state updates from agents"""
        if "custom_data" in updates:
            self.custom_data.update(updates["custom_data"])
            del updates["custom_data"]
        
        self._state.update(updates)
    
    def _get_agent_data(self, agent):
        """Get agent-specific data"""
        return f"data_for_{id(agent)}" if agent else "general_data"
```

### Adding Agent Configurations

1. **Create YAML configuration:**

```yaml
# config/my_agent_config.yaml
name: MyAgent
version: 1.0.0
description: My custom agent configuration

system_prompts:
  default: |
    You are MyAgent, a specialized assistant for [your domain].
    Focus on [specific capabilities].

capabilities:
  max_iterations: 100
  supported_formats:
    - json
    - yaml
    - csv

behaviors:
  auto_save: true
  error_recovery: true
  
templates:
  response_format: |
    Status: {status}
    Result: {result}
    Next Action: {next_action}
```

2. **Load configuration in your agent:**

```python
from core.template_loader import QirkPromptTemplate

class MyAgent(Agent):
    def __init__(self, lm: LM, environment=None):
        super().__init__(lm, environment)
        self.config = QirkPromptTemplate.load("config/my_agent_config.yaml")
        
        # Use config values
        self.max_iterations = self.config.get("capabilities", {}).get("max_iterations", 50)
        system_prompt = self.config.get("system_prompts", {}).get("default", "")
```

### Adding Custom Tools

Tools are methods decorated with `@tool` that agents can call:

```python
@tool
def search_database(self, query: str, limit: int = 10) -> str:
    """
    Search the database with the given query.
    
    Args:
        query: Search query string
        limit: Maximum number of results
    """
    # Your tool implementation
    results = self.database.search(query, limit=limit)
    return f"Found {len(results)} results for '{query}'"

@tool
def send_notification(self, message: str, priority: str = "normal") -> str:
    """Send a notification to the user."""
    self.notification_service.send(message, priority)
    return f"Notification sent: {message}"
```

### Adding Custom Scorers

Scorers measure specific aspects of agent/environment behavior:

```python
# scorer/my_custom_scorer.py
from zagency.core.scorer import AgentScorer, EnvironmentScorer
from typing import Dict, Any, Optional

class CustomAgentScorer(AgentScorer):
    """Scores agents based on custom metrics"""
    
    def score(self, agent: Agent, step_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Access agent internals
        custom_metric = agent._internal_state.get("my_metric", 0)
        
        # Calculate score (0.0 to 1.0)
        score = min(custom_metric / 100.0, 1.0)
        
        result = {
            "score": score,
            "custom_metric": custom_metric,
            "scorer": self.name
        }
        
        # Track for aggregation
        self._scores.append(result)
        return result

class CustomEnvironmentScorer(EnvironmentScorer):
    """Scores environment based on state changes"""
    
    def score(self, environment: Environment, agent: Optional[Agent] = None) -> Dict[str, Any]:
        state = environment.get_state(agent)
        
        # Example: score based on task completion
        completed_tasks = state.get("completed_tasks", [])
        total_tasks = state.get("total_tasks", 1)
        
        score = len(completed_tasks) / total_tasks
        
        result = {
            "score": score,
            "completed": len(completed_tasks),
            "total": total_tasks
        }
        
        self._scores.append(result)
        return result
```

### Creating Evaluations

Build comprehensive evaluations with multiple scorers:

```python
# evaluations/my_evaluation.py
from zagency.core.evaluation import Evaluation, EvaluationSuite
from zagency.handler.step_handler import StepHandler

def create_coding_evaluation():
    """Create an evaluation for coding tasks"""
    
    # Define scorers
    scorers = [
        ThinkingTokenScorer(),
        FileAccessScorer(),
        CompletionTimeScorer(),
        TestPassRateScorer()  # Custom scorer
    ]
    
    # Create evaluation
    evaluation = Evaluation(
        name="coding_benchmark",
        scorers=scorers,
        aggregation_fn=lambda results: sum(r["completed"] for r in results)
    )
    
    return evaluation

# Run evaluation suite
def run_evaluation_suite():
    suite = EvaluationSuite("comprehensive_eval")
    
    # Add multiple evaluations
    suite.add_evaluation(create_coding_evaluation())
    suite.add_evaluation(create_debugging_evaluation())
    
    # Define agents and datasets
    agents = [QirkTheCoder(lm), SimpleCoderAgent(lm)]
    datasets = {
        "easy_tasks": load_dataset("data/easy_tasks.json"),
        "hard_tasks": load_dataset("data/hard_tasks.json")
    }
    
    # Run suite
    results = suite.run(
        agents=agents,
        environment_class=CodingEnvironment,
        datasets=datasets,
        max_steps_per_datum=100
    )
    
    # Print summary
    print(suite.summarize())
```

### Creating Test Scenarios

1. **Add scenario YAML file:**

```yaml
# qirk_tests/scenarios/my_test_scenario.yaml
environment:
  metadata:
    name: "My Test Scenario"
    description: "Test my agent's capabilities"
    difficulty: "medium"
    tags: ["custom", "feature"]
  
  initial_state:
    data_to_process: ["item1", "item2", "item3"]
    expected_output: "processed_items"
  
  expectations:
    custom_checks_pass: true
    final_state_contains:
      - key: "processing_complete"
        value: true
    files_should_exist:
      - "output.json"
```

2. **Run your scenarios:**

```bash
# Run specific scenario
python qirk_tests/test_runner.py qirk_tests/scenarios/my_test_scenario.yaml

# Run all scenarios with custom tag
python qirk_tests/test_runner.py --tag custom
```

## 🛠️ Available Agents

### QirkTheCoder (`agents/qirk_coder.py`)
Advanced coding agent with capabilities:
- File reading/writing with context awareness
- Patch-based editing for precise changes
- Test execution and failure handling
- Git integration and version control
- Multi-language support

**Usage:**
```python
from agents.qirk_coder import QirkTheCoder
from environments.coding_environment import CodingEnvironment

env = CodingEnvironment(project_root="./my_project")
agent = QirkTheCoder(lm, environment=env)
```

### GrepAgent (`grepAgent/grep_agent.py`)
Specialized for finding function definitions in codebases:
- Bash command execution
- Pattern-based searching across languages
- Confidence scoring for matches
- Repository cloning and navigation

**Usage:**
```python
from grepAgent.grep_agent import GrepAgent
from grepAgent.grep_environment import FileSystemEnvironment

env = FileSystemEnvironment(repo_url="https://github.com/user/repo.git")
agent = GrepAgent(lm, env)
```

### Example Agents (`agents/example_agents.py`)
- **FileProcessorAgent**: Processes files from a queue
- **CollaborativeAgent**: Multi-agent task coordination

## 📊 Testing & Evaluation

The framework includes comprehensive testing and evaluation capabilities:

### Evaluation Framework (`core/evaluation.py`)
- **Scorer System**: Modular scoring components for different aspects
  - `AgentScorer`: Evaluates agent internals (tokens, decisions, etc.)
  - `EnvironmentScorer`: Evaluates environment state changes
  - `TraceScorer`: Evaluates complete interaction traces
- **Evaluation Engine**: Run agents on datasets with multiple scorers
- **Evaluation Suites**: Compare multiple agents across datasets
- **Custom Aggregation**: Flexible result aggregation functions

**Example evaluation:**
```python
from zagency.core import Evaluation, ThinkingTokenScorer, CompletionTimeScorer

# Create evaluation with scorers
eval = Evaluation(
    name="performance_eval",
    scorers=[ThinkingTokenScorer(), CompletionTimeScorer()]
)

# Run on dataset
results = eval.run(
    agent=my_agent,
    environment_class=MyEnvironment,
    dataset=test_cases,
    max_steps_per_datum=50
)

print(f"Average completion rate: {results['aggregate_scores']['completion_rate']:.2%}")
```

### QirkTheCoder Testing (`qirk_tests/`)
- YAML-based scenario definitions
- Automated test execution
- Performance metrics tracking
- Multi-scenario evaluation

**Run tests:**
```bash
# All scenarios
python qirk_tests/test_runner.py

# Specific scenario
python qirk_tests/test_runner.py qirk_tests/scenarios/bug_fix_zero_division.yaml

# With custom model
python qirk_tests/test_runner.py --model gpt-4-turbo --max-steps 30
```

### GrepAgent Evaluation (`grepAgent/`)
- Function finding accuracy testing
- Command efficiency measurement
- Dataset-based evaluation

**Run evaluation:**
```bash
# Single function search
python grepAgent/run_grep_task.py --function "my_func" --repo "github.com/user/repo"

# Dataset evaluation
python grepAgent/run_grep_task.py --dataset grepAgent/example_dataset.json
```

## 🔗 Integration Examples

### Multi-Agent Collaboration

```python
from handler.step_handler import MultiAgentOrchestrator

# Create orchestrator
orchestrator = MultiAgentOrchestrator()
orchestrator.add_agent(planner_agent, role="Planner")
orchestrator.add_agent(coder_agent, role="Coder") 
orchestrator.add_agent(tester_agent, role="Tester")

# Run collaborative session
result = orchestrator.run_collaboration(
    initial_task="Implement user authentication",
    max_steps=100
)
```

### Custom Handler

```python
class CustomHandler(StepHandler):
    def should_continue(self) -> bool:
        """Custom termination logic"""
        state = self.environment.get_state()
        return not state.get("task_complete", False)
    
    def pre_step_hook(self, agent: Agent):
        """Called before each agent step"""
        self.console.print(f"[blue]Executing {agent.__class__.__name__}")
    
    def post_step_hook(self, agent: Agent, result: Dict[str, Any]):
        """Called after each agent step"""
        if result.get("status") == "error":
            self.console.print(f"[red]Error in {agent.__class__.__name__}")
```

## 📚 Advanced Features

### Context Management
Automatic trajectory compression and token optimization:

```python
# Configure in YAML
context_management:
  max_trajectory_length: 50
  compression_threshold: 30000
  preserve_recent_messages: 10
  summarization_model: "gpt-3.5-turbo"
```

### Error Handling
Built-in retry and rollback mechanisms:

```python
behaviors:
  error_handling:
    max_retries: 3
    rollback_on_test_failure: true
    log_errors: true
```

### Token Usage Tracking
Monitor LM API usage across agents:

```python
result = agent.invoke(messages)
if "token_usage" in result:
    print(f"Tokens used: {result['token_usage']}")
```

## 🚦 Best Practices

1. **Start Simple**: Begin with basic agents and gradually add complexity
2. **State Management**: Use environments for all shared data
3. **Tool Design**: Keep tools focused and well-documented
4. **Testing**: Create scenarios for every new feature
5. **Evaluation**: Design scorers that measure meaningful metrics
   - Use `AgentScorer` for internal agent metrics
   - Use `EnvironmentScorer` for state change analysis
   - Use `TraceScorer` for end-to-end performance
6. **Configuration**: Use YAML configs for agent behavior
7. **Error Handling**: Implement robust error recovery
8. **Step Granularity**: Keep steps atomic and observable
9. **Benchmarking**: Create evaluation suites to compare agent variants

## 🤝 Contributing

1. **Add New Agents**: Follow the patterns in `agents/example_agents.py`
2. **Extend Environments**: Inherit from base environment classes  
3. **Create Tests**: Add scenarios to `qirk_tests/scenarios/`
4. **Documentation**: Update READMEs for new components
5. **Configuration**: Use YAML for agent configurations

## 📖 Further Reading

- [`NEW_FRAMEWORK.md`](NEW_FRAMEWORK.md) - Detailed framework documentation
- [`qirk_tests/README.md`](qirk_tests/README.md) - Testing framework guide
- [`grepAgent/README.md`](grepAgent/README.md) - GrepAgent specific documentation
- [`Thesis.md`](Thesis.md) - Theoretical foundations

---


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
