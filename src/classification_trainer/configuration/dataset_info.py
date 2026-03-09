"""Dataset metadata definitions and YAML loader."""

from __future__ import annotations

import re
from collections.abc import Iterator

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

from classification_trainer.utils.common_paths import CommonPaths

from ._hf_validators import validate_hf_name

_COLUMN_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")
_SPLIT_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")

_TRAINING_COLUMN_NAME: str = "training"
_PREDICTION_COLUMN_NAME: str = "prediction"
_EVAL_INSTRUCTIONS_COLUMN_NAME: str = "eval_instructions"

_STR_LABELS_COLUMN_NAME: str = "string_labels"
_TRAINING_SPLIT_NAME: str = "train"
_TEST_SPLIT_NAME: str = "test"
_VALIDATION_SPLIT_NAME: str = "validation"
_CLASSIFICATION_RESULT_COLUMN_NAME: str = "classification_result"

_TOKENIZED_TRAINING_COLUMN_NAME = "input_ids"
_TOKENIZED_TRAINING_ATTENTION_MASK_COLUMN_NAME = "attention_mask"
_MASKED_TOKENIZED_TRAINING_COLUMN_NAME = "labels"
_TOKENIZED_EVAL_COLUMN_NAME = "eval_input_ids"
_TOKENIZED_EVAL_ATTENTION_MASK_COLUMN_NAME = "eval_attention_mask"


class DatasetInfo(BaseModel):
    """Validated metadata for a HuggingFace dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str
    hugging_face_user_name: str
    # huggingface_name: str
    content_column_name: str
    label_column_name: str
    new_column_prefix: str
    # Used to train the model.
    training_split_name: str
    # Used for evaluation after training is complete.
    test_split_name: str | None = None
    # Used for early stopping and evaluation during training.
    validation_split_name: str | None = None
    # Maximum rows to load per split. -1 means load all rows.
    max_rowcount: int = -1
    # Evaluation settings
    positive_case: str
    case_invariant_comparison: bool = True
    search_length: int = -1
    search_from_end: bool = False
    # Sequence lengths to check coverage for in analyze-dataset.
    potential_sequence_lengths: list[int] = [1024, 1536, 2048]

    def get_generated_column_names(self) -> Iterator[str]:
        yield self.classification_result_column_name
        yield self.prediction_column_name
        yield self.string_labels_column_name
        yield self.evaluation_instructions_column_name
        yield self.training_column_name
        yield self.tokenized_training_column_name
        yield self.tokenized_training_attention_maske_column_name
        yield self.masked_tokenized_training_column_name
        yield self.tokenized_eval_column_name
        yield self.tokenized_eval_attention_maske_column_name

    @property
    def huggingface_name(self) -> str:
        return self.hugging_face_user_name + "/" + self.dataset_name

    @property
    def masked_tokenized_training_column_name(sefl) -> str:
        return _MASKED_TOKENIZED_TRAINING_COLUMN_NAME

    @property
    def tokenized_eval_attention_maske_column_name(self) -> str:
        return _TOKENIZED_EVAL_ATTENTION_MASK_COLUMN_NAME

    @property
    def tokenized_eval_column_name(self) -> str:
        return _TOKENIZED_EVAL_COLUMN_NAME

    @property
    def tokenized_training_attention_maske_column_name(self) -> str:
        return _TOKENIZED_TRAINING_ATTENTION_MASK_COLUMN_NAME

    @property
    def tokenized_training_column_name(self) -> str:
        return _TOKENIZED_TRAINING_COLUMN_NAME

    @property
    def classification_result_column_name(self) -> str:
        return self.new_column_prefix + _CLASSIFICATION_RESULT_COLUMN_NAME

    @property
    def evaluation_instructions_column_name(self) -> str:
        return self.new_column_prefix + _EVAL_INSTRUCTIONS_COLUMN_NAME

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

    @field_validator("max_rowcount")
    @classmethod
    def validate_max_rowcount(cls, v: int) -> int:
        if v != -1 and v <= 0:
            raise ValueError(f"max_rowcount must be -1 (load all) or a positive integer, got {v}")
        return v

    @field_validator("training_split_name", "test_split_name", "validation_split_name")
    @classmethod
    def validate_split_name(cls, v: str | None) -> str | None:
        if v is not None and not _SPLIT_NAME_RE.match(v):
            raise ValueError(f"Invalid split name '{v}': must be non-empty and contain only [a-zA-Z0-9_-]")
        return v

    @field_validator("potential_sequence_lengths")
    @classmethod
    def validate_sequence_lengths(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("potential_sequence_lengths must contain at least one value")
        for val in v:
            if val <= 0:
                raise ValueError(f"potential_sequence_lengths values must be positive integers, got {val}")
        # Deduplicate while preserving order
        seen: set[int] = set()
        deduped: list[int] = []
        for val in v:
            if val not in seen:
                seen.add(val)
                deduped.append(val)
        return deduped

    def add_split(self) -> DatasetInfo:
        if self.validation_split_name or self.test_split_name:
            raise ValueError("Dataset already contains test and evaluation splits.")
        return self.model_copy(
            update={
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
    yaml_path = CommonPaths.get().dataset_info / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset info file not found: {yaml_path}")
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return DatasetInfo(**data)
