import pytest
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from datetime import datetime
from zagency.environments.coding_environment import FileState, CodingEnvironment


class TestFileState:
    """Test the FileState class."""
    
    def test_filestate_init(self):
        """Test FileState initialization."""
        fs = FileState("test.py", "original content")
        
        assert fs.path == "test.py"
        assert fs.original_content == "original content"
        assert fs.current_content == "original content"
        assert fs.history == []
        assert isinstance(fs.last_modified, datetime)
    
    def test_filestate_init_empty(self):
        """Test FileState initialization with empty content."""
        fs = FileState("test.py")
        
        assert fs.path == "test.py"
        assert fs.original_content == ""
        assert fs.current_content == ""
    
    def test_filestate_update(self):
        """Test updating FileState content."""
        fs = FileState("test.py", "original")
        initial_time = fs.last_modified
        
        fs.update("new content", "Added feature")
        
        assert fs.current_content == "new content"
        assert fs.original_content == "original"  # Should not change
        assert len(fs.history) == 1
        assert fs.last_modified > initial_time
        
        history_entry = fs.history[0]
        assert history_entry["previous_content"] == "original"
        assert history_entry["new_content"] == "new content"
        assert history_entry["description"] == "Added feature"
        assert isinstance(history_entry["timestamp"], datetime)
    
    def test_filestate_multiple_updates(self):
        """Test multiple updates to FileState."""
        fs = FileState("test.py", "v1")
        
        fs.update("v2", "Update 1")
        fs.update("v3", "Update 2")
        
        assert fs.current_content == "v3"
        assert fs.original_content == "v1"
        assert len(fs.history) == 2
        
        assert fs.history[0]["previous_content"] == "v1"
        assert fs.history[0]["new_content"] == "v2"
        assert fs.history[1]["previous_content"] == "v2"
        assert fs.history[1]["new_content"] == "v3"
    
    def test_filestate_get_diff(self):
        """Test getting diff between original and current content."""
        original = "line 1\nline 2\nline 3"
        fs = FileState("test.py", original)
        fs.update("line 1\nmodified line 2\nline 3", "Modified line 2")
        
        diff = fs.get_diff()
        
        assert "test.py" in diff
        assert "-line 2" in diff
        assert "+modified line 2" in diff
    
    def test_filestate_get_diff_no_changes(self):
        """Test getting diff when there are no changes."""
        fs = FileState("test.py", "same content")
        
        diff = fs.get_diff()
        
        # Should be empty string for no differences
        assert diff == ""
    
    def test_filestate_rollback_single_step(self):
        """Test rolling back a single step."""
        fs = FileState("test.py", "v1")
        fs.update("v2", "Update 1")
        fs.update("v3", "Update 2")
        
        assert fs.current_content == "v3"
        
        fs.rollback(1)
        
        assert fs.current_content == "v2"
        assert len(fs.history) == 1
    
    def test_filestate_rollback_multiple_steps(self):
        """Test rolling back multiple steps."""
        fs = FileState("test.py", "v1")
        fs.update("v2", "Update 1")
        fs.update("v3", "Update 2")
        fs.update("v4", "Update 3")
        
        fs.rollback(2)
        
        assert fs.current_content == "v2"
        assert len(fs.history) == 1
    
    def test_filestate_rollback_too_many_steps(self):
        """Test rolling back more steps than available."""
        fs = FileState("test.py", "v1")
        fs.update("v2", "Update 1")
        
        fs.rollback(5)  # More than available
        
        # Should rollback all available steps but not further
        assert fs.current_content == "v1"  # Should go back to original
        assert len(fs.history) == 0


class TestCodingEnvironment:
    """Test the CodingEnvironment class."""
    
    def test_coding_environment_init(self, temp_project_dir):
        """Test CodingEnvironment initialization."""
        env = CodingEnvironment(str(temp_project_dir))
        
        # Use resolve() on both paths to handle symlinks (like /private vs /var on macOS)
        assert env.project_root.resolve() == temp_project_dir.resolve()
        assert env.file_states == {}
        assert env.test_results == []
        assert env.pending_patches == []
        
        state = env._state
        # Use resolve() to handle symlinks (like /private vs /var on macOS)
        assert Path(state["project_root"]).resolve() == temp_project_dir.resolve()
        assert state["modified_files"] == []
        assert state["test_status"] == "unknown"
        assert state["pending_tasks"] == []
        assert state["completed_tasks"] == []
    
    def test_coding_environment_init_default(self):
        """Test CodingEnvironment initialization with default project root."""
        with patch('zagency.environments.coding_environment.Path') as mock_path:
            mock_path.return_value.resolve.return_value = "/current/dir"
            env = CodingEnvironment()
            assert str(env.project_root) == "/current/dir"
    
    def test_get_state(self, temp_project_dir):
        """Test getting environment state."""
        env = CodingEnvironment(str(temp_project_dir))
        
        # Add some file states
        fs1 = FileState("file1.py", "content1")
        fs1.update("new content1", "change")
        env.file_states["file1.py"] = fs1
        
        fs2 = FileState("file2.py", "content2")
        env.file_states["file2.py"] = fs2
        
        state = env.get_state()
        
        assert "file_summary" in state
        assert "pending_patches_count" in state
        assert state["pending_patches_count"] == 0
        
        file_summary = state["file_summary"]
        assert "file1.py" in file_summary
        assert "file2.py" in file_summary
        assert file_summary["file1.py"]["modified"] is True
        assert file_summary["file1.py"]["changes"] == 1
        assert file_summary["file2.py"]["modified"] is False
        assert file_summary["file2.py"]["changes"] == 0
    
    def test_load_file_existing(self, sample_files):
        """Test loading an existing file."""
        env = CodingEnvironment(str(sample_files["test_file"].parent))
        
        file_state = env.load_file("test_file.py")
        
        assert isinstance(file_state, FileState)
        assert file_state.path == "test_file.py"
        assert "def hello():" in file_state.current_content
        assert "test_file.py" in env.file_states
    
    def test_load_file_nonexistent(self, temp_project_dir):
        """Test loading a nonexistent file."""
        env = CodingEnvironment(str(temp_project_dir))
        
        file_state = env.load_file("new_file.py")
        
        assert isinstance(file_state, FileState)
        assert file_state.path == "new_file.py"
        assert file_state.current_content == ""
        assert file_state.original_content == ""
    
    def test_load_file_security_check(self, temp_project_dir):
        """Test that loading files outside project root is prevented."""
        env = CodingEnvironment(str(temp_project_dir))
        
        with pytest.raises(ValueError, match="outside project root"):
            env.load_file("../../../etc/passwd")
    
    def test_load_file_cached(self, sample_files):
        """Test that loading the same file twice returns cached version."""
        env = CodingEnvironment(str(sample_files["test_file"].parent))
        
        file_state1 = env.load_file("test_file.py")
        file_state2 = env.load_file("test_file.py")
        
        assert file_state1 is file_state2  # Should be same object
    
    def test_save_file(self, temp_project_dir):
        """Test saving a file to disk."""
        env = CodingEnvironment(str(temp_project_dir))
        
        # Load and modify a file
        file_state = env.load_file("new_file.py")
        file_state.update("print('hello world')", "Added print statement")
        
        result = env.save_file("new_file.py")
        
        assert result is True
        
        # Check file was actually written
        saved_file = temp_project_dir / "new_file.py"
        assert saved_file.exists()
        assert saved_file.read_text() == "print('hello world')"
        
        # Check state was updated
        state = env.get_state()
        assert "new_file.py" in state["modified_files"]
    
    def test_save_file_with_directories(self, temp_project_dir):
        """Test saving a file in nested directories."""
        env = CodingEnvironment(str(temp_project_dir))
        
        file_state = env.load_file("src/utils/helper.py")
        file_state.update("# Helper functions", "Added comment")
        
        result = env.save_file("src/utils/helper.py")
        
        assert result is True
        
        # Check directories were created
        saved_file = temp_project_dir / "src" / "utils" / "helper.py"
        assert saved_file.exists()
        assert saved_file.read_text() == "# Helper functions"
    
    def test_save_file_not_loaded(self, temp_project_dir):
        """Test saving a file that wasn't loaded."""
        env = CodingEnvironment(str(temp_project_dir))
        
        result = env.save_file("nonexistent.py")
        
        assert result is False
    
    def test_create_patch(self, temp_project_dir):
        """Test creating a patch."""
        env = CodingEnvironment(str(temp_project_dir))
        
        original = "line 1\nline 2\nline 3"
        modified = "line 1\nmodified line 2\nline 3"
        
        patch = env.create_patch("test.py", original, modified)
        
        assert "test.py" in patch
        assert "-line 2" in patch
        assert "+modified line 2" in patch
    
    @patch('zagency.environments.coding_environment.subprocess.run')
    def test_apply_patch_success(self, mock_run, temp_project_dir):
        """Test successfully applying a patch."""
        # Setup mock subprocess
        mock_run.return_value.returncode = 0
        mock_run.return_value.stderr = ""
        
        env = CodingEnvironment(str(temp_project_dir))
        
        # Create a file to patch
        test_file = temp_project_dir / "test.py"
        test_file.write_text("original content")
        
        # Load file into environment
        file_state = env.load_file("test.py")
        
        # We need to mock the file operations within apply_patch
        original_open = open
        
        def mock_open_context(file_path, mode='r', *args, **kwargs):
            if 'w' in mode and '.patch_' in str(file_path):
                # Writing patch file - use real file operation
                return original_open(file_path, mode, *args, **kwargs)
            elif 'r' in mode and str(file_path).endswith('test.py'):
                # Reading the file after patch - return patched content
                return mock_open(read_data="patched content").return_value
            else:
                # Default behavior
                return original_open(file_path, mode, *args, **kwargs)
        
        with patch("builtins.open", side_effect=mock_open_context):
            success, message = env.apply_patch("test.py", "dummy patch")
        
        assert success is True
        assert "successfully" in message.lower()
        assert file_state.current_content == "patched content"
    
    @patch('zagency.environments.coding_environment.subprocess.run')
    def test_apply_patch_failure(self, mock_run, temp_project_dir):
        """Test patch application failure."""
        # Setup mock subprocess to fail
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "patch failed"
        
        env = CodingEnvironment(str(temp_project_dir))
        file_state = env.load_file("test.py")
        
        success, message = env.apply_patch("test.py", "invalid patch")
        
        assert success is False
        assert "patch failed" in message.lower()
    
    @patch('zagency.environments.coding_environment.subprocess.run')
    def test_run_tests_pytest_success(self, mock_run, temp_project_dir):
        """Test running tests with pytest successfully."""
        # Setup pytest directory
        (temp_project_dir / "tests").mkdir()
        
        # Mock successful test run
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test output"
        mock_run.return_value.stderr = ""
        
        env = CodingEnvironment(str(temp_project_dir))
        result = env.run_tests()
        
        assert result["success"] is True
        assert result["command"] == "pytest"
        assert result["stdout"] == "test output"
        assert len(env.test_results) == 1
        assert env._state["test_status"] == "passed"
    
    @patch('zagency.environments.coding_environment.subprocess.run')
    def test_run_tests_custom_command(self, mock_run, temp_project_dir):
        """Test running tests with custom command."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "custom test output"
        mock_run.return_value.stderr = ""
        
        env = CodingEnvironment(str(temp_project_dir))
        result = env.run_tests("python -m unittest")
        
        assert result["command"] == "python -m unittest"
        # Check call was made with correct parameters (using resolve to handle symlinks)
        call_args = mock_run.call_args
        assert call_args[1]["shell"] is True
        assert call_args[1]["capture_output"] is True  
        assert call_args[1]["text"] is True
        assert call_args[1]["timeout"] == 60
        assert call_args[0][0] == "python -m unittest"
        # Use resolve() to handle path differences like /private/var vs /var
        assert call_args[1]["cwd"].resolve() == temp_project_dir.resolve()
    
    @patch('zagency.environments.coding_environment.subprocess.run')
    def test_run_tests_failure(self, mock_run, temp_project_dir):
        """Test running tests that fail."""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "test failures"
        
        env = CodingEnvironment(str(temp_project_dir))
        result = env.run_tests("pytest")
        
        assert result["success"] is False
        assert result["stderr"] == "test failures"
        assert env._state["test_status"] == "failed"
    
    @patch('zagency.environments.coding_environment.subprocess.run')
    def test_run_tests_timeout(self, mock_run, temp_project_dir):
        """Test running tests that timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired("pytest", 60)
        
        env = CodingEnvironment(str(temp_project_dir))
        result = env.run_tests("pytest")
        
        assert result["success"] is False
        assert "timed out" in result["error"].lower()
        assert env._state["test_status"] == "timeout"
    
    @patch('zagency.environments.coding_environment.subprocess.run')
    def test_run_tests_exception(self, mock_run, temp_project_dir):
        """Test running tests with unexpected exception."""
        mock_run.side_effect = Exception("Unexpected error")
        
        env = CodingEnvironment(str(temp_project_dir))
        result = env.run_tests("pytest")
        
        assert result["success"] is False
        assert "Unexpected error" in result["error"]
        assert env._state["test_status"] == "error"
    
    def test_get_project_structure(self, sample_files):
        """Test getting project structure."""
        env = CodingEnvironment(str(sample_files["test_file"].parent))
        
        structure = env.get_project_structure()
        
        assert structure["type"] == "directory"
        assert structure["name"] == sample_files["test_file"].parent.name
        assert "children" in structure
        
        # Should contain our test files
        file_names = [child["name"] for child in structure["children"] if child["type"] == "file"]
        assert "test_file.py" in file_names
    
    def test_get_project_structure_max_depth(self, temp_project_dir):
        """Test project structure with depth limit."""
        # Create deep directory structure
        deep_dir = temp_project_dir / "level1" / "level2" / "level3" / "level4"
        deep_dir.mkdir(parents=True)
        (deep_dir / "deep_file.py").write_text("deep content")
        
        env = CodingEnvironment(str(temp_project_dir))
        structure = env.get_project_structure(max_depth=2)
        
        # Should not include level4 due to depth limit
        def find_deep_file(node):
            if node["type"] == "file" and node["name"] == "deep_file.py":
                return True
            if node["type"] == "directory" and "children" in node:
                return any(find_deep_file(child) for child in node["children"])
            return False
        
        assert not find_deep_file(structure)
    
    def test_get_project_structure_ignores_common_dirs(self, temp_project_dir):
        """Test that project structure ignores common directories."""
        # Create ignored directories
        for ignored_dir in ['.git', '__pycache__', 'node_modules', '.env', '.venv']:
            (temp_project_dir / ignored_dir).mkdir()
            (temp_project_dir / ignored_dir / "file.txt").write_text("content")
        
        env = CodingEnvironment(str(temp_project_dir))
        structure = env.get_project_structure()
        
        # Should not contain ignored directories
        child_names = [child["name"] for child in structure["children"]]
        for ignored_dir in ['.git', '__pycache__', 'node_modules', '.env', '.venv']:
            assert ignored_dir not in child_names 