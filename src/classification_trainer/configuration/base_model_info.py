"""Base model definitions and YAML loader.

Only instruct models are supported. See .research/spec_decisions.md.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from ._hf_validators import validate_hf_name
from .chat_template_info import ChatTemplateInfo, load_chat_template_info


class BaseModelInfo(BaseModel):
    """Validated configuration for a base model (instruct models only)."""

    model_config = ConfigDict(frozen=True)

    huggingface_name: str
    chat_template: str

    @field_validator("huggingface_name")
    @classmethod
    def validate_huggingface_name(cls, v: str) -> str:
        return validate_hf_name(v)

    @field_validator("chat_template")
    @classmethod
    def validate_chat_template(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("chat_template must be a non-empty string")
        return v

    @property
    def chat_template_info(self) -> ChatTemplateInfo:
        return load_chat_template_info(self.chat_template)


def load_base_model_info(name: str) -> BaseModelInfo:
    """Load a BaseModelInfo from a YAML file in the base_model_info directory.

    Args:
        name: The yaml filename without extension (e.g. "qwen2.5-1.5b-instruct").

    Returns:
        A validated BaseModelInfo instance.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
        pydantic.ValidationError: If the file data fails validation.
    """
    from classification_trainer.utils.common_paths import CommonPaths

    yaml_path = CommonPaths.get().base_model_info / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Base model info file not found: {yaml_path}")
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return BaseModelInfo(**data)
