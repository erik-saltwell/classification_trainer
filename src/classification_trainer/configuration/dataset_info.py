"""Dataset metadata definitions and YAML loader."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from ._hf_validators import validate_hf_name

DATASET_INFO_DIR = Path("dataset_info")

_COLUMN_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")
_SPLIT_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

_TRAINING_COLUMN_NAME: str = "training"
_PREDICTION_COLUMN_NAME: str = "prediction"
_STR_LABELS_COLUMN_NAME: str = "string_labels"
_TRAINING_SPLIT_NAME: str = "train"
_TEST_SPLIT_NAME: str = "test"
_VALIDATION_SPLIT_NAME: str = "validation"


class DatasetInfo(BaseModel):
    """Validated metadata for a HuggingFace dataset."""

    model_config = ConfigDict(frozen=True)

    huggingface_name: str
    content_column_name: str
    label_column_name: str
    new_column_prefix: str
    is_split: bool = False
    # Used to train the model.
    training_split_name: str | None = None
    # Used for evaluation after training is complete.
    test_split_name: str | None = None
    # Used for early stopping and evaluation during training.
    validation_split_name: str | None = None

    @property
    def training_column_name(self) -> str:
        return self.new_column_prefix + _TRAINING_COLUMN_NAME

    @property
    def prediction_column_name(self) -> str:
        return self.new_column_prefix + _PREDICTION_COLUMN_NAME

    @property
    def string_labels_column_name(self) -> str:
        return self.new_column_prefix + _STR_LABELS_COLUMN_NAME

    @field_validator("huggingface_name")
    @classmethod
    def validate_huggingface_name(cls, v: str) -> str:
        return validate_hf_name(v)

    @field_validator("content_column_name", "label_column_name", "new_column_prefix")
    @classmethod
    def validate_column_name(cls, v: str) -> str:
        if not _COLUMN_NAME_RE.match(v):
            raise ValueError(f"Invalid column name '{v}': must be non-empty and contain only [a-zA-Z0-9_]")
        return v

    @field_validator("training_split_name", "test_split_name", "validation_split_name")
    @classmethod
    def validate_split_name(cls, v: str | None) -> str | None:
        if v is not None and not _SPLIT_NAME_RE.match(v):
            raise ValueError(f"Invalid split name '{v}': must be non-empty and contain only [a-zA-Z0-9_-]")
        return v

    def add_split(self) -> DatasetInfo:
        return self.model_copy(
            update={
                "is_split": True,
                "training_split_name": _TRAINING_SPLIT_NAME,
                "test_split_name": _TEST_SPLIT_NAME,
                "validation_split_name": _VALIDATION_SPLIT_NAME,
            }
        )


def load_dataset_info(name: str) -> DatasetInfo:
    """Load a DatasetInfo from a YAML file in the dataset_info directory.

    Args:
        name: The yaml filename without extension (e.g. "reddit-rpg").

    Returns:
        A validated DatasetInfo instance.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
        pydantic.ValidationError: If the file data fails validation.
    """
    yaml_path = DATASET_INFO_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset info file not found: {yaml_path}")
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return DatasetInfo(**data)
