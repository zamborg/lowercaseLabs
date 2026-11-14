import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from pydantic import BaseModel
from zagency.core.template_loader import TemplateLoader, QirkPromptTemplate


class TestTemplateLoader:
    """Test the TemplateLoader base class."""
    
    def test_load_yaml_valid_file(self, temp_project_dir):
        """Test loading a valid YAML file."""
        yaml_file = temp_project_dir / "test.yaml"
        yaml_content = """
        key1: value1
        key2: 42
        nested:
          inner_key: inner_value
        """
        yaml_file.write_text(yaml_content)
        
        result = TemplateLoader.load_yaml(str(yaml_file))
        expected = {
            "key1": "value1",
            "key2": 42,
            "nested": {"inner_key": "inner_value"}
        }
        assert result == expected
    
    def test_load_yaml_empty_file(self, temp_project_dir):
        """Test loading an empty YAML file."""
        yaml_file = temp_project_dir / "empty.yaml"
        yaml_file.write_text("")
        
        result = TemplateLoader.load_yaml(str(yaml_file))
        assert result == {}
    
    def test_load_yaml_nonexistent_file(self):
        """Test loading a nonexistent YAML file."""
        with pytest.raises(FileNotFoundError):
            TemplateLoader.load_yaml("nonexistent.yaml")
    
    def test_load_from_env(self):
        """Test loading values from environment variables."""
        # Create a mock Template class with fields
        class MockTemplate(BaseModel):
            field1: str = "default1"
            field2: int = 10
            field3: str = "default3"
        
        class MockLoader(TemplateLoader):
            Template = MockTemplate
        
        # Set some environment variables
        with patch.dict(os.environ, {"FIELD1": "env_value1", "FIELD2": "20"}):
            result = MockLoader.load_from_env(MockTemplate.model_fields)
            
            assert result == {"field1": "env_value1", "field2": "20"}
            # FIELD3 not set, so shouldn't be in result
            assert "field3" not in result
    
    def test_load_with_yaml_only(self, temp_project_dir):
        """Test loading with only YAML data."""
        class TestTemplate(BaseModel):
            name: str = "default"
            count: int = 1
        
        class TestLoader(TemplateLoader):
            Template = TestTemplate
        
        yaml_file = temp_project_dir / "config.yaml"
        yaml_file.write_text("name: yaml_name\ncount: 5")
        
        result = TestLoader.load(str(yaml_file))
        assert result.name == "yaml_name"
        assert result.count == 5
    
    def test_load_with_env_override(self, temp_project_dir):
        """Test loading with environment variables overriding YAML."""
        class TestTemplate(BaseModel):
            name: str = "default"
            count: int = 1
        
        class TestLoader(TemplateLoader):
            Template = TestTemplate
        
        yaml_file = temp_project_dir / "config.yaml"
        yaml_file.write_text("name: yaml_name\ncount: 5")
        
        with patch.dict(os.environ, {"NAME": "env_name"}):
            result = TestLoader.load(str(yaml_file))
            assert result.name == "env_name"  # ENV overrides YAML
            assert result.count == 5  # YAML value
    
    def test_load_with_explicit_overrides(self, temp_project_dir):
        """Test loading with explicit overrides."""
        class TestTemplate(BaseModel):
            name: str = "default"
            count: int = 1
            enabled: bool = False
        
        class TestLoader(TemplateLoader):
            Template = TestTemplate
        
        yaml_file = temp_project_dir / "config.yaml"
        yaml_file.write_text("name: yaml_name\ncount: 5\nenabled: true")
        
        with patch.dict(os.environ, {"NAME": "env_name"}):
            result = TestLoader.load(str(yaml_file), name="override_name", count=10)
            assert result.name == "override_name"  # Explicit override wins
            assert result.count == 10  # Explicit override wins
            assert result.enabled is True  # YAML value
    
    def test_load_without_yaml(self):
        """Test loading without YAML file (defaults and env only)."""
        class TestTemplate(BaseModel):
            name: str = "default"
            count: int = 1
        
        class TestLoader(TemplateLoader):
            Template = TestTemplate
        
        with patch.dict(os.environ, {"COUNT": "42"}):
            result = TestLoader.load()
            assert result.name == "default"  # Pydantic default
            assert result.count == 42  # ENV value
    
    def test_load_validation_error(self, temp_project_dir):
        """Test that Pydantic validation errors are raised."""
        class TestTemplate(BaseModel):
            count: int  # Required field, no default
        
        class TestLoader(TemplateLoader):
            Template = TestTemplate
        
        yaml_file = temp_project_dir / "config.yaml"
        yaml_file.write_text("name: some_name")  # Missing required 'count'
        
        with pytest.raises(Exception):  # Pydantic validation error
            TestLoader.load(str(yaml_file))


class TestQirkPromptTemplate:
    """Test the QirkPromptTemplate class."""
    
    def test_qirk_template_fields(self):
        """Test that QirkPromptTemplate has the expected fields."""
        template = QirkPromptTemplate.Template
        fields = template.model_fields
        
        expected_fields = {
            "system_prompt",
            "memory_prompt", 
            "trajectory_compression_prompt",
            "task_template_prompt"
        }
        
        assert set(fields.keys()) == expected_fields
        
        # All fields should be strings
        for field_name, field_info in fields.items():
            assert field_info.annotation == str
    
    def test_qirk_template_loading(self, temp_project_dir):
        """Test loading a QirkPromptTemplate from YAML."""
        yaml_file = temp_project_dir / "prompts.yaml"
        yaml_content = """
        system_prompt: "You are a helpful assistant"
        memory_prompt: "Remember the context"
        trajectory_compression_prompt: "Compress this trajectory"
        task_template_prompt: "Task: {task}, Trajectory: {trajectory}"
        """
        yaml_file.write_text(yaml_content)
        
        result = QirkPromptTemplate.load(str(yaml_file))
        assert result.system_prompt == "You are a helpful assistant"
        assert result.memory_prompt == "Remember the context"
        assert result.trajectory_compression_prompt == "Compress this trajectory"
        assert result.task_template_prompt == "Task: {task}, Trajectory: {trajectory}" 