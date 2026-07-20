import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_lm():
    """Mock LM for testing agents."""
    lm = Mock()
    lm.model = "test-model"
    lm.invoke.return_value = {
        "response": Mock(choices=[Mock(message=Mock(tool_calls=None))]),
        "usage": {"total_tokens": 100}
    }
    return lm

@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_files(temp_project_dir):
    """Create sample files in temp directory for testing."""
    # Create a simple Python file
    test_file = temp_project_dir / "test_file.py"
    test_file.write_text("def hello():\n    return 'world'\n")
    
    # Create a nested directory with file
    nested_dir = temp_project_dir / "src"
    nested_dir.mkdir()
    nested_file = nested_dir / "main.py"
    nested_file.write_text("from test_file import hello\n\nif __name__ == '__main__':\n    print(hello())\n")
    
    return {
        "test_file": test_file,
        "nested_file": nested_file
    } 