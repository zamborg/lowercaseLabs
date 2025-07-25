import pytest
from unittest.mock import Mock, patch, MagicMock
from zagency.handler.step_handler import StepHandler, MultiAgentOrchestrator
from zagency.core.environment import SharedEnvironment
from zagency.core.agent import Agent


class MockAgent(Agent):
    """Mock agent for testing StepHandler."""
    
    def __init__(self, lm, environment, name="TestAgent"):
        super().__init__(lm, environment)
        self.name = name
        self.step_count = 0
        self.should_error = False
        self.status = "active"
    
    def step(self, environment):
        self.step_count += 1
        if self.should_error:
            raise Exception(f"Error in {self.name}")
        
        return {
            "status": self.status,
            "step_count": self.step_count,
            "message": f"{self.name} executed step {self.step_count}"
        }
    
    def ingest_state(self, environment):
        return environment.get_state()
    
    def synthesize_lm_input(self, **kwargs):
        return [{"role": "user", "content": "test"}]


class TestStepHandler:
    """Test the StepHandler class."""
    
    def test_step_handler_init(self):
        """Test StepHandler initialization."""
        handler = StepHandler()
        
        assert isinstance(handler.environment, SharedEnvironment)
        assert handler.agents == []
        assert handler.step_count == 0
    
    def test_step_handler_init_with_environment(self):
        """Test StepHandler initialization with custom environment."""
        custom_env = SharedEnvironment()
        handler = StepHandler(custom_env)
        
        assert handler.environment is custom_env
    
    def test_add_agent(self, mock_lm):
        """Test adding an agent to the handler."""
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment)
        
        handler.add_agent(agent)
        
        assert len(handler.agents) == 1
        assert agent in handler.agents
        assert agent.environment is handler.environment
        assert agent in handler.environment._agents
    
    def test_add_agent_duplicate(self, mock_lm):
        """Test adding the same agent twice."""
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment)
        
        handler.add_agent(agent)
        handler.add_agent(agent)  # Add same agent twice
        
        assert len(handler.agents) == 1  # Should not duplicate
    
    def test_remove_agent(self, mock_lm):
        """Test removing an agent from the handler."""
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment)
        
        handler.add_agent(agent)
        assert len(handler.agents) == 1
        
        handler.remove_agent(agent)
        assert len(handler.agents) == 0
        assert agent not in handler.environment._agents
    
    def test_remove_agent_not_present(self, mock_lm):
        """Test removing an agent that's not in the handler."""
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment)
        
        # Should not error when removing agent that's not present
        handler.remove_agent(agent)
        assert len(handler.agents) == 0
    
    @patch('zagency.handler.step_handler.Console')
    def test_step_single_agent(self, mock_console_class, mock_lm):
        """Test executing a step with a single agent."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment, "TestAgent")
        handler.add_agent(agent)
        
        results = handler.step()
        
        assert handler.step_count == 1
        assert "MockAgent_0" in results
        result = results["MockAgent_0"]
        assert result["status"] == "active"
        assert result["step_count"] == 1
        assert "TestAgent executed step 1" in result["message"]
    
    @patch('zagency.handler.step_handler.Console')
    def test_step_multiple_agents(self, mock_console_class, mock_lm):
        """Test executing a step with multiple agents."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        handler = StepHandler()
        agent1 = MockAgent(mock_lm, handler.environment, "Agent1")
        agent2 = MockAgent(mock_lm, handler.environment, "Agent2")
        
        handler.add_agent(agent1)
        handler.add_agent(agent2)
        
        results = handler.step()
        
        assert len(results) == 2
        assert "MockAgent_0" in results
        assert "MockAgent_1" in results
        assert results["MockAgent_0"]["message"] == "Agent1 executed step 1"
        assert results["MockAgent_1"]["message"] == "Agent2 executed step 1"
    
    @patch('zagency.handler.step_handler.Console')
    def test_step_with_agent_error(self, mock_console_class, mock_lm):
        """Test executing a step when an agent throws an error."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment, "ErrorAgent")
        agent.should_error = True
        handler.add_agent(agent)
        
        results = handler.step()
        
        assert "MockAgent_0" in results
        result = results["MockAgent_0"]
        assert result["status"] == "error"
        assert "Error in ErrorAgent" in result["error"]
    
    @patch('zagency.handler.step_handler.Console')
    def test_run_until_all_idle(self, mock_console_class, mock_lm):
        """Test running until all agents are idle."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment)
        agent.status = "idle"  # Agent will be idle immediately
        handler.add_agent(agent)
        
        handler.run(max_steps=10)
        
        # Should stop after first step because agent is idle
        assert handler.step_count == 1
    
    @patch('zagency.handler.step_handler.Console')
    def test_run_max_steps(self, mock_console_class, mock_lm):
        """Test running until max steps reached."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment)
        agent.status = "active"  # Agent will never be idle
        handler.add_agent(agent)
        
        handler.run(max_steps=3)
        
        # Should stop after 3 steps
        assert handler.step_count == 3
    
    @patch('zagency.handler.step_handler.Console')
    def test_run_with_stop_condition(self, mock_console_class, mock_lm):
        """Test running with a custom stop condition."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        handler = StepHandler()
        agent = MockAgent(mock_lm, handler.environment)
        handler.add_agent(agent)
        
        # Stop condition that triggers after 2 steps
        def stop_condition(state):
            return handler.step_count >= 2
        
        handler.run(max_steps=10, stop_condition=stop_condition)
        
        # Should stop after 2 steps due to stop condition
        assert handler.step_count == 2


class TestMultiAgentOrchestrator:
    """Test the MultiAgentOrchestrator class."""
    
    @patch('zagency.handler.step_handler.Console')
    def test_orchestrator_init(self, mock_console_class):
        """Test MultiAgentOrchestrator initialization."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        orchestrator = MultiAgentOrchestrator()
        
        assert isinstance(orchestrator.handler, StepHandler)
        assert orchestrator.console is mock_console
    
    @patch('zagency.handler.step_handler.Console')
    def test_orchestrator_init_with_environment(self, mock_console_class):
        """Test MultiAgentOrchestrator initialization with custom environment."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        custom_env = SharedEnvironment()
        orchestrator = MultiAgentOrchestrator(custom_env)
        
        assert orchestrator.handler.environment is custom_env
    
    @patch('zagency.handler.step_handler.Console')
    def test_add_agent_with_role(self, mock_console_class, mock_lm):
        """Test adding an agent with a role description."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        orchestrator = MultiAgentOrchestrator()
        agent = MockAgent(mock_lm, orchestrator.handler.environment)
        
        orchestrator.add_agent(agent, role="Code Editor")
        
        assert len(orchestrator.handler.agents) == 1
        assert agent in orchestrator.handler.agents
        mock_console.print.assert_called_with("[green]Added MockAgent as Code Editor[/green]")
    
    @patch('zagency.handler.step_handler.Console')
    def test_add_agent_without_role(self, mock_console_class, mock_lm):
        """Test adding an agent without a role description."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        orchestrator = MultiAgentOrchestrator()
        agent = MockAgent(mock_lm, orchestrator.handler.environment)
        
        orchestrator.add_agent(agent)
        
        assert len(orchestrator.handler.agents) == 1
        # Should not call print for role when role is None
        mock_console.print.assert_not_called()
    
    @patch('zagency.handler.step_handler.Console')
    def test_run_collaboration(self, mock_console_class, mock_lm):
        """Test running a collaboration session."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        orchestrator = MultiAgentOrchestrator()
        agent = MockAgent(mock_lm, orchestrator.handler.environment)
        agent.status = "idle"  # Will stop quickly
        orchestrator.add_agent(agent)
        
        initial_state = {"task": "test_task", "files_to_process": []}
        orchestrator.run_collaboration(initial_state, max_steps=5)
        
        # Check that initial state was set
        env_state = orchestrator.handler.environment.get_state()
        assert env_state["task"] == "test_task"
        assert env_state["files_to_process"] == []
    
    @patch('zagency.handler.step_handler.Console')
    def test_run_collaboration_stop_condition(self, mock_console_class, mock_lm):
        """Test that collaboration stops when stop condition is met."""
        mock_console = Mock()
        mock_console_class.return_value = mock_console
        
        orchestrator = MultiAgentOrchestrator()
        agent = MockAgent(mock_lm, orchestrator.handler.environment)
        orchestrator.add_agent(agent)
        
        # Set up state that will trigger stop condition
        initial_state = {
            "files_to_process": [],  # Empty files list
            "task_queue": {}  # Empty task queue
        }
        
        orchestrator.run_collaboration(initial_state, max_steps=10)
        
        # Should stop early due to stop condition
        assert orchestrator.handler.step_count <= 10 