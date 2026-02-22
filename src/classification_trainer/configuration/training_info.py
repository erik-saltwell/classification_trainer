from __future__ import annotations

from enum import StrEnum

import yaml
from pydantic import BaseModel

from classification_trainer.utils import FragmentID, get_fragment


class TrainingLengthType(StrEnum):
    STEPS = "steps"
    EPOCHS = "epoch"


class TrainingInfo(BaseModel):
    model_name: str  # must be valid hugging face name, use hf_validators.py
    hugging_face_user_name: str  # must be valid hugging face name, use hf_validators.py
    system_prompt_name: FragmentID
    training_length_type: TrainingLengthType
    training_length: float
    max_sequence_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    train_on_outputs_only: bool = True
    dtype: str | None = None
    load_in_4bit: bool = True
    packing: bool = False
    seed: int = 3412

    @property
    def hugging_face_model_name(self) -> str:
        return self.hugging_face_user_name + "/" + self.model_name

    @property
    def system_prompt(self) -> str:
        return get_fragment(self.system_prompt_name)


def load_training_info(name: str) -> TrainingInfo:
    """Load a TrainingInfo from a YAML file in the training_info directory.

    Args:
        name: The yaml filename without extension (e.g. "my-training-run").

    Returns:
        A validated TrainingInfo instance.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
        pydantic.ValidationError: If the file data fails validation.
    """
    from classification_trainer.utils.common_paths import CommonPaths

    yaml_path = CommonPaths.get().training_info / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Training info file not found: {yaml_path}")
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return TrainingInfo(**data)
