from typing import Type, Any, Dict
from pydantic import BaseModel
import os
import yaml

class TemplateLoader:
    """
    Base class for loading configuration templates with Pydantic validation.
    Loads values in the following precedence:
        1. YAML file
        2. Environment variable (if set)
        3. Code default (Pydantic default)
    Subclass this with a Pydantic model as the `Template` inner class.
    """
    Template: Type[BaseModel] = BaseModel  # Should be overridden by subclasses

    @classmethod
    def load_yaml(cls, yaml_path: str) -> Dict[str, Any]:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return data

    @classmethod
    def load_from_env(cls, fields: Dict[str, Any]) -> Dict[str, Any]:
        env_data = {}
        for field in fields:
            env_var = field.upper()
            if env_var in os.environ:
                env_data[field] = os.environ[env_var]
        return env_data

    @classmethod
    def load(cls, yaml_path: str = None, **overrides) -> BaseModel:
        # 1. Load from YAML if provided
        yaml_data = cls.load_yaml(yaml_path) if yaml_path else {}
        # 2. Load from environment variables
        env_data = cls.load_from_env(cls.Template.model_fields)
        # 3. Merge: YAML < ENV < explicit overrides
        merged = {**yaml_data, **env_data, **overrides}
        # 4. Validate and instantiate the template
        return cls.Template(**merged)


class QirkPromptTemplate(TemplateLoader):
    class Template(BaseModel):
        system_prompt: str
        memory_prompt: str
        trajectory_compression_prompt: str # this gets fed to a *smaller* model to compress the trajectory
        task_template_prompt: str # This is a prompt that contains fstring references to the task and the trajectory