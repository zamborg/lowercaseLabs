#!/usr/bin/env python3
"""
Unit tests for QirkTheCoder tools.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.base import LiteLLM
from agents.qirk_coder import QirkTheCoder
from environments.coding_environment import CodingEnvironment


class TestQirkTools:
    """Test individual QirkTheCoder tools."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def qirk_agent(self, temp_project):
        """Create a QirkTheCoder instance with test environment."""
        env = CodingEnvironment(temp_project)
        lm = LiteLLM(model="gpt-3.5-turbo")  # Use cheaper model for tests
        return QirkTheCoder(lm, temp_project, env)
    
    def test_read_file(self, qirk_agent, temp_project):
        """Test read_file tool."""
        # Create test file
        test_file = Path(temp_project) / "test.py"
        test_content = "def hello():\n    print('Hello, World!')"
        test_file.write_text(test_content)
        
        # Test reading entire file
        result = qirk_agent.read_file("test.py")
        assert "hello()" in result
        assert "Hello, World!" in result
        assert "1 |" in result  # Check line numbers
        
        # Test reading with line range
        result = qirk_agent.read_file("test.py", start_line=2, end_line=2)
        assert "print('Hello, World!')" in result
        assert "2 |" in result
        
        # Test reading non-existent file
        result = qirk_agent.read_file("nonexistent.py")
        assert "Error" in result
        assert "does not exist" in result
    
    def test_write_file(self, qirk_agent, temp_project):
        """Test write_file tool."""
        # Test creating new file
        content = "# New file\nprint('Created by test')"
        result = qirk_agent.write_file("new_file.py", content)
        assert "Successfully wrote" in result
        
        # Verify file was created
        file_path = Path(temp_project) / "new_file.py"
        assert file_path.exists()
        assert file_path.read_text() == content
        
        # Test overwriting file
        new_content = "# Updated\nprint('Updated')"
        result = qirk_agent.write_file("new_file.py", new_content)
        assert "Successfully wrote" in result
        assert file_path.read_text() == new_content
    
    def test_search_files(self, qirk_agent, temp_project):
        """Test search_files tool."""
        # Create test files
        (Path(temp_project) / "file1.py").write_text("def process_data():\n    return 42")
        (Path(temp_project) / "file2.py").write_text("def process_info():\n    return 'data'")
        (Path(temp_project) / "readme.md").write_text("This project processes data")
        
        # Test basic search
        result = qirk_agent.search_files("process")
        assert "file1.py" in result
        assert "file2.py" in result
        assert "readme.md" in result
        
        # Test with file pattern
        result = qirk_agent.search_files("process", file_pattern="*.py")
        assert "file1.py" in result
        assert "file2.py" in result
        assert "readme.md" not in result
        
        # Test no matches
        result = qirk_agent.search_files("nonexistent_pattern")
        assert "No matches found" in result
    
    def test_create_patch(self, qirk_agent, temp_project):
        """Test patch creation."""
        old_content = "def add(a, b):\n    return a + b"
        new_content = "def add(a, b):\n    # Add two numbers\n    return a + b"
        
        result = qirk_agent.create_patch_from_edit("math.py", old_content, new_content)
        assert "Created patch" in result
        assert "+    # Add two numbers" in result
        assert "@@" in result  # Unified diff format
    
    def test_run_command(self, qirk_agent, temp_project):
        """Test run_command tool."""
        # Test simple command
        result = qirk_agent.run_command("echo 'Hello from test'")
        assert "Hello from test" in result
        assert "Exit code: 0" in result
        
        # Test command with error
        result = qirk_agent.run_command("false")  # Command that returns non-zero
        assert "Exit code: 1" in result
        
        # Test timeout
        result = qirk_agent.run_command("sleep 5", timeout=1)
        assert "timed out" in result
    
    def test_view_project_structure(self, qirk_agent, temp_project):
        """Test view_project_structure tool."""
        # Create directory structure
        (Path(temp_project) / "src").mkdir()
        (Path(temp_project) / "src" / "main.py").write_text("# Main")
        (Path(temp_project) / "tests").mkdir()
        (Path(temp_project) / "tests" / "test_main.py").write_text("# Test")
        (Path(temp_project) / "README.md").write_text("# Project")
        
        result = qirk_agent.view_project_structure()
        assert "Project structure:" in result
        assert "src" in result
        assert "main.py" in result
        assert "tests" in result
        assert "test_main.py" in result
        assert "README.md" in result
    
    def test_git_operations(self, qirk_agent, temp_project):
        """Test git-related tools."""
        # Initialize git repo
        os.system(f"cd {temp_project} && git init")
        
        # Create and add a file
        test_file = Path(temp_project) / "test.txt"
        test_file.write_text("Initial content")
        
        # Test git status
        result = qirk_agent.git_status()
        assert "Untracked: test.txt" in result
        
        # Stage file
        result = qirk_agent.git_add(["test.txt"])
        assert "Staged 1 file(s)" in result
        
        # Test git status after staging
        result = qirk_agent.git_status()
        assert "Added: test.txt" in result or "new file" in result
        
        # Commit
        result = qirk_agent.git_commit("Initial commit", "Added test file")
        assert "Created commit" in result
        
        # Test git log
        result = qirk_agent.git_log(max_entries=5)
        assert "Initial commit" in result


class TestQirkEnvironmentIntegration:
    """Test QirkTheCoder with CodingEnvironment integration."""
    
    @pytest.fixture
    def env_and_agent(self):
        """Create environment and agent for testing."""
        temp_dir = tempfile.mkdtemp()
        env = CodingEnvironment(temp_dir)
        lm = LiteLLM(model="gpt-3.5-turbo")
        agent = QirkTheCoder(lm, temp_dir, env)
        
        yield env, agent, temp_dir
        
        shutil.rmtree(temp_dir)
    
    def test_file_state_tracking(self, env_and_agent):
        """Test that file modifications are tracked in environment."""
        env, agent, temp_dir = env_and_agent
        
        # Write a file
        agent.write_file("tracked.py", "print('original')")
        
        # Check environment state
        state = env.get_state()
        assert "tracked.py" in state["modified_files"]
        
        # Modify file
        agent.write_file("tracked.py", "print('modified')")
        
        # Check file state
        file_state = env.file_states["tracked.py"]
        assert file_state.current_content == "print('modified')"
        assert len(file_state.history) == 1
    
    def test_test_execution(self, env_and_agent):
        """Test running tests through environment."""
        env, agent, temp_dir = env_and_agent
        
        # Create a simple test file
        test_code = """
def test_pass():
    assert 1 + 1 == 2

def test_fail():
    assert 1 + 1 == 3
"""
        agent.write_file("test_example.py", test_code)
        
        # Run tests
        result = agent.run_tests("python -m pytest test_example.py")
        
        # Check result
        assert "FAILED" in result
        assert "test_fail" in result
        
        # Check environment state
        state = env.get_state()
        assert state["test_status"] == "failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])