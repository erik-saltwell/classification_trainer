"""Target model definitions, output paths, and registry lookup."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class TargetModelName(StrEnum):
    """Identifiers for fine-tuned target models."""

    REDDIT_RPG_POST_CLASSIFICATION = "reddit_rpg_post_classification"
    IMDB_TEST = "imdb_sentiment_test"
    NONE = "none"


@dataclass
class TargetModelInfo:
    """Metadata for a fine-tuned target model.

    Attributes:
        output_directory: Local path where the trained model artifacts are stored.
        hf_name: HuggingFace repository name for the target model.
    """

    output_directory: Path
    hf_name: str


_targeet_model_registry: dict[TargetModelName, TargetModelInfo] = {
    TargetModelName.REDDIT_RPG_POST_CLASSIFICATION: TargetModelInfo(
        output_directory=Path(""), hf_name=TargetModelName.REDDIT_RPG_POST_CLASSIFICATION
    ),
    TargetModelName.IMDB_TEST: TargetModelInfo(output_directory=Path(""), hf_name=TargetModelName.IMDB_TEST),
}


def get_base_model_info(model_name: TargetModelName) -> TargetModelInfo:
    """Look up the metadata for a target model.

    Args:
        model_name: The target model identifier to look up.

    Returns:
        The corresponding TargetModelInfo with output path and HuggingFace name.

    Raises:
        KeyError: If the model name is not registered.
    """
    if model_name not in _targeet_model_registry:
        raise KeyError(f"No model data registered for {model_name}")
    return _targeet_model_registry[model_name]
