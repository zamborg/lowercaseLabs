# The zagency Thesis: Towards Environment-Centric Agent Evaluation

## Abstract

The rapid advancement of AI agents has outpaced our ability to meaningfully evaluate their capabilities. Current evaluation methodologies suffer from static benchmarks, isolated task assessment, and a fundamental misalignment between how we test agents and how they operate in practice. This thesis presents a novel framework for agent evaluation centered on **environmental state management** and **declarative scenario specification**, arguing that the future of agent assessment lies not in what agents can do in isolation, but how they interact with and transform their environments over time.

## The Problem with Current Agent Evaluation

### 1. The Benchmark Trap

Traditional AI evaluation relies heavily on static benchmarks - fixed datasets with predetermined correct answers. While this approach works for models, it fails catastrophically for agents because:

- **Agents are stateful**: They maintain context, learn from interactions, and modify their environment
- **Agents are compositional**: Their value emerges from tool use, planning, and multi-step reasoning
- **Agents are contextual**: Performance depends heavily on environmental conditions and constraints

Consider SWE-bench, where success is measured by test passage. This misses crucial aspects: Did the agent understand the codebase? Did it make maintainable changes? Did it consider edge cases beyond the tests?

### 2. The Tool-Centric Fallacy

Most agent frameworks focus on tools as the primary abstraction:
```python
agent.use_tool("write_file", {"path": "main.py", "content": "..."})
```

This is backwards. Tools are merely the mechanism; the environment is the substrate. A file system, a codebase, a conversation - these are environments with state that agents navigate and transform.

### 3. The Measurement Gap

Current metrics focus on final outcomes (task completion rate, accuracy) while ignoring the journey:
- How many steps did it take?
- What intermediate states were created?
- How robust is the solution to perturbations?
- Could another agent continue from this state?

## The zagency Framework: A New Paradigm

### Core Principle: Environment as First-Class Citizen

```python
class Environment(ABC):
    """Environments hold state that agents transform."""
    
    @abstractmethod
    def get_state(self, agent: Optional["Agent"] = None) -> Dict[str, Any]:
        """State is readable - agents observe their world."""
        pass
    
    @abstractmethod
    def update_state(self, updates: Dict[str, Any], agent: Optional["Agent"] = None):
        """State is mutable - agents change their world."""
        pass
```

Environments are not just containers for files or data. They are:
- **Living systems** with internal consistency rules
- **Observable** by agents who need to understand current state
- **Mutable** through agent actions
- **Shareable** across multiple agents for collaboration

### The Step Function: Discrete Agent Evolution

```python
class Agent(ABC):
    @abstractmethod
    def step(self, environment: Environment) -> Dict[str, Any]:
        """One discrete step of agent evolution."""
        # 1. Observe environment
        # 2. Decide on action  
        # 3. Execute action
        # 4. Return step summary
```

The step function makes agent behavior:
- **Inspectable**: Each step can be logged, replayed, analyzed
- **Interruptible**: Execution can pause for human review
- **Composable**: Steps can be chained, parallelized, or modified
- **Measurable**: Metrics can be computed per-step, not just per-task

### Declarative Evaluation Scenarios

```yaml
environment:
  metadata:
    name: "Fix Authentication Bug"
    description: "Agent must diagnose and fix a subtle authentication bypass"
    
  initial_state:
    files:
      - path: "auth.py"
        content: |
          def check_auth(token):
              # Bug: timing attack vulnerability
              if len(token) != 32:
                  return False
              for i, char in enumerate(token):
                  if char != VALID_TOKEN[i]:
                      return False
              return True
    
  expectations:
    security_fixed:
      - "constant-time comparison implemented"
      - "no timing attack possible"
    tests_pass: true
    code_quality:
      - "maintains readability"
      - "includes security comment"
```

Evaluation scenarios as YAML enable:
- **Reproducibility**: Same initial conditions every time
- **Composability**: Mix and match scenario components
- **Accessibility**: Non-programmers can create evaluations
- **Versioning**: Track how evaluations evolve over time

## Novel Contributions

### 1. State-Diff Evaluation

Traditional: "Did the agent complete the task?"
zagency: "How did the environment state evolve?"

```python
def evaluate_trajectory(states: List[EnvironmentState]) -> Metrics:
    # Measure state evolution entropy
    # Detect unnecessary state mutations
    # Identify state regression
    # Calculate state stability
```

### 2. Multi-Agent Evaluation Scenarios

Environments naturally support multiple agents:
```yaml
agents:
  - name: "Developer"
    role: "implement_feature"
  - name: "Reviewer"  
    role: "review_and_suggest"
  - name: "Tester"
    role: "write_tests"

success_criteria:
  - "feature implemented"
  - "review feedback addressed"
  - "comprehensive tests added"
```

This enables evaluation of:
- Collaborative capability
- Communication through shared state
- Handoff quality
- Emergent team dynamics

### 3. Continuous Evaluation Spaces

Instead of binary pass/fail, evaluate on continuous metrics:

```python
class EvaluationSpace:
    dimensions = {
        "correctness": (0.0, 1.0),      # Did it work?
        "efficiency": (0.0, 1.0),        # How quickly?
        "robustness": (0.0, 1.0),        # How thoroughly?
        "elegance": (0.0, 1.0),          # How cleanly?
        "maintainability": (0.0, 1.0),   # How sustainable?
    }
```

Agents can be plotted in this space, revealing:
- Trade-offs between dimensions
- Pareto frontiers of agent capabilities
- Specialization patterns

### 4. Evaluation Scenario Generation

Scenarios themselves can be generated:
```python
def generate_scenario(complexity: float, domain: str) -> Scenario:
    # Use LLMs to create novel, balanced scenarios
    # Ensure solvability through symbolic verification
    # Inject controlled difficulties
    # Generate corresponding evaluations
```

This creates an ever-expanding evaluation suite that:
- Prevents overfitting to fixed benchmarks
- Explores edge cases systematically
- Adapts to agent improvements

## Implications for Agent Development

### 1. Development Becomes Experimentation

With zagency, agent development shifts from "implement features" to "evolve behaviors":
- Run agent against scenario suite
- Identify failure patterns
- Modify agent architecture
- Measure improvement across evaluation space

### 2. Interpretability Through Environments

Understanding agents becomes understanding their environmental interactions:
- What state changes do they make?
- What patterns emerge across scenarios?
- How do they handle unexpected states?

### 3. Robustness Through Variation

Agents must handle:
- **State perturbations**: Slightly modified initial conditions
- **Scenario mutations**: Variations on core tasks
- **Environmental drift**: States that change during execution

## Future Directions

### 1. Learned Evaluation Functions

Train models to predict human judgment of agent trajectories:
```python
evaluator = LearnedEvaluator.from_human_feedback(
    trajectories_with_ratings
)
score = evaluator.evaluate(agent_trajectory)
```

### 2. Adversarial Scenario Generation

Create scenarios specifically designed to expose agent weaknesses:
```python
scenario = AdversarialGenerator.create_scenario(
    target_agent=agent,
    difficulty="maximize",
    domain="security"
)
```

### 3. Meta-Evaluation

Evaluate the evaluation framework itself:
- Are scenarios predictive of real-world performance?
- Do metrics correlate with human preferences?
- How do we validate evaluation validity?

## Conclusion: A Call to Action

The future of AI agents depends on our ability to meaningfully evaluate their capabilities. The zagency framework offers a path forward:

1. **Embrace environments** as the primary abstraction for agent interaction
2. **Adopt step-based execution** for fine-grained observability
3. **Use declarative scenarios** for reproducible, shareable evaluation
4. **Measure state evolution**, not just task completion
5. **Build continuous evaluation spaces** that capture nuanced performance

The agent revolution is not about creating more powerful tools - it's about understanding how intelligent systems interact with and transform their worlds. By putting environments at the center of our evaluation methodology, we can build agents that are not just capable, but comprehensible, controllable, and collaborative.

The code is written. The framework is ready. The question now is: How will you evaluate your agents?

---

*"The best way to predict the future is to invent it. The best way to evaluate agents is to give them worlds to transform."*

## References and Further Reading

1. **Environment-Centric Design Patterns**
   - The Actor Model (Hewitt, 1973) - Agents as environmental actors
   - Ecological Psychology (Gibson, 1979) - Affordances in agent environments
   - Situated Cognition (Lave, 1988) - Context-dependent intelligence

2. **State-Based Evaluation Theory**
   - Model Checking (Clarke et al., 1999) - Formal verification of state spaces
   - Reinforcement Learning (Sutton & Barto, 2018) - Reward from state transitions
   - Process Mining (van der Aalst, 2016) - Learning from execution traces

3. **Multi-Agent Systems**
   - Collaborative Intelligence (Malone, 2018) - Human-AI collaboration patterns
   - Swarm Intelligence (Bonabeau et al., 1999) - Emergent behaviors
   - Game Theory (Myerson, 1991) - Strategic interactions in shared environments

4. **Practical Applications**
   - SWE-bench (2024) - Software engineering benchmarks
   - WebArena (2024) - Web navigation environments  
   - GAIA (2023) - General AI assistant benchmarks

---

*The zagency Framework is open source and available at [github.com/lowercaseLabs/zagency](https://github.com/lowercaseLabs/zagency)*

# Zubin Addendums:

The tight philosophy of evaluations and agent-interactions is as follows:

### Primitives:
- Agents
- Environments
- Runners (aka: handlers)
- LMs
  - this doesn't have its own section but they are implemented as some baseLM that has an invoke method. 
  - A litellm wrapper is implemented for ease.
  - Implements marshaling/unmarshalling helper functions (eg: litellm.Message -> openai_message)

### Environments
Environments offer a substrate upon which agents run. Agents are distinct from LM's with Tool-Calling *because* they run ontop of an environment. In our topology, Agent's contain an `environment` object with which they interact with through `tools`. 
It is the responsibility of the environment to expose an API for which agents can interact with it. For example, a simple file-system environment might have three functions:
### `FileSystemEnvironment`
- `read_file`
- `write_file`
- `list_files`

This would serve the basis for an agent being able to interact with some underlying filesystem. Where these commands are run, how they are executed in the undelrying environment, and all other idiosynchracies of specific methods are masked by this API. 

Environments also have a fundamental `loading` method that instantiates them from a given *state*. This is to say that an environment has a `_state: Dict[str, Any]` object that models the internal environment. *NOTE*: this means that an environment that exposes some `bash` like interface is not, in and of itself, the `bash` interface. Rather it is the *exposure* layer for the agent <> `bash` interaction. As such, there is stil some fundamental *fabric* on which the `Environment` runs. 

The `state` for an environment is useful as it allows the `agent` to *percieve* it's surroundings. Furthermore, for evaluation, the environment is provided some *initial* state (consider a :seed:) and this provides the basis for the `agent`. EG: In `SWEBench` the `state` is a checkout of a git repo: `git_repo, git_hash` + some `gh-issue-text: [issue, helper_text]`. The environment might offer helper functions like `execute_pytest` ... but the initial conditions can be read from a single dictionary and instantiated. It is the responsibility of the `Environment` to instantiate the repo, and the responsibility of the `Agent` to grab it's task from the `Environment`.

### Agents

Agents are *monoliths* in our universe. The design of this system is an `Agent` class that implements a `step` fuction as well as a variety of tools. Developers can mutate the `step` function however they see fit, as well as implement a `StepRunner` in whatever order they care about, but ultimately the natural use case is as follows: `percieve(env)` -> `invoke_lm(perception + agent_state)` -> `?(execute tool calls)` -> `update(agent_state + actions)`

Agent's interact with an `environment` that they have access to on construction, the `environment` exposes an API for them to interact with, and the `agent` implements `tools` that can be executed by the agent's `LM`. By default, the agent has access to all tools on every invocation with the decorator `@tool` for any given function. 

Creating a new agent involves three things:

1. Writing a series of self-functions decorated with `@tool` and these can use the `self.environment` class of the Agent
2. Writing an `invoke` method, or using the default (that uses `self._history` as a message dictionary alongside all tools). By default, state is encoded in `_history` and overriding the `get_history()` method can enable complicated interactions. `_history` (if left for default) is an OAI style message list
3. Writing a `step` method. This is what the `StepRunner` will sequence until `exit()` is called by the agent, triggering an end to the `step-runner's` loop.

This might seem like a lot of code to write, but the goal of this framework is to induce *solidity* and *observability*, not to reduce code writing. This framework is also very clear, leading LM's like Claude-Code to operate really effectively when building subclasses, increasing experimentation velocity. 

### Runners

This is the simplest primative of this framework. This is disambiguated from the agent just for code clarity. Runners can be as complicated as necessary, but they expose the final API between an agent and environment + starting conditions. 

The job of a runner is to take an environment and agent pair (composable) and execute until a certain condition is reached. Functors can be written to measure the state of the environment and the agent and inform stopping conditions. By default the agent maintains `is_complete` as a member variable, and stopping on `is_complete` happens when the agent calls it's `exit()` tool. 

Runner's don't store information or offer multiple state interactions, they only execute agents in a for loop, or in parallel as designed. This interface is (ex)portable such that it can interface with `weave.Evaluation` objects and adapters are written to map a `dataset` into a runner invocation. Runner's also operate on `single-env<agent>` pairs, which is to say that a runner is offered a starting condition for an env and kicks off `<agents>` in said environment. This can then be mapped to a full dataset of evaluation calls etc.


## Okay so why do we care about this?

We want to build the primitives and infrastrucutre to enable the development and experimentation of self-improving agents. To do so we must formalize a methedology of *evaluation*

