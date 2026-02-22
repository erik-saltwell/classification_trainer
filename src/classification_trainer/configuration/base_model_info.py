"""Base model definitions and YAML loader."""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from classification_trainer.utils.text_fragments import FragmentID

from .chat_template_info import ChatTemplateName


class BaseModelInfo(BaseModel):
    """Validated configuration for a base model.

    For instruct models: chat_template must be set (not NONE); fragment IDs must be NONE.
    For non-instruct models: chat_template must be NONE; both fragment IDs must be set (not NONE).
    """

    model_config = ConfigDict(frozen=True)

    is_instruct: bool
    chat_template: ChatTemplateName = ChatTemplateName.NONE
    training_fragment_id: FragmentID = FragmentID.NONE
    eval_fragment_id: FragmentID = FragmentID.NONE

    @model_validator(mode="after")
    def validate_instruct_consistency(self) -> BaseModelInfo:
        if self.is_instruct:
            if self.chat_template == ChatTemplateName.NONE:
                raise ValueError("chat_template cannot be NONE for instruct models")
            if self.training_fragment_id != FragmentID.NONE:
                raise ValueError("training_fragment_id must be NONE for instruct models")
            if self.eval_fragment_id != FragmentID.NONE:
                raise ValueError("eval_fragment_id must be NONE for instruct models")
        else:
            if self.chat_template != ChatTemplateName.NONE:
                raise ValueError("chat_template must be NONE for non-instruct models")
            if self.training_fragment_id == FragmentID.NONE:
                raise ValueError("training_fragment_id cannot be NONE for non-instruct models")
            if self.eval_fragment_id == FragmentID.NONE:
                raise ValueError("eval_fragment_id cannot be NONE for non-instruct models")
        return self


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
