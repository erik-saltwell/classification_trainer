from __future__ import annotations

from enum import StrEnum

import yaml
from pydantic import BaseModel, ConfigDict

from classification_trainer.utils.common_paths import CommonPaths


class SaveFormat(StrEnum):
    """Internal slugs used for format directories and HuggingFace repo suffixes.

    All formats — including GGUF — use a single slug. Multiple GGUF quantization
    levels are stored as separate files (``<model>-gguf-<quant>.gguf``) inside one
    shared ``gguf/`` directory and published to one shared HuggingFace repository.
    """

    GGUF = "gguf"
    LORA = "lora"
    MERGED = "merged"


class PublishingInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str

    # GGUF: one directory and one HuggingFace repo is created per quantization.
    # Slug pattern: gguf-<quant>  (e.g. output_models/<model>/gguf-q8_0/)
    # Only used when save_gguf or publish_gguf is True.
    gguf_quantizations: list[str] = ["q8_0"]

    # Unsloth save method for the merged checkpoint format.
    merged_save_method: str = "merged_16bit"

    # --- Save to local disk flags ---
    save_gguf: bool = False
    save_lora: bool = False
    save_merged: bool = False

    # --- Publish to HuggingFace Hub flags ---
    publish_gguf: bool = False
    publish_lora: bool = False
    publish_merged: bool = False

    @property
    def any_save_enabled(self) -> bool:
        return any([self.save_gguf, self.save_lora, self.save_merged])

    @property
    def any_publish_enabled(self) -> bool:
        return any([self.publish_gguf, self.publish_lora, self.publish_merged])


def load_publishing_info(name: str) -> PublishingInfo:
    """Load a PublishingInfo from a YAML file in the publishing_info directory.

    Args:
        name: The yaml filename without extension (e.g. "my-run").

    Returns:
        A validated PublishingInfo instance.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
        pydantic.ValidationError: If the file data fails validation.
    """
    yaml_path = CommonPaths.get().publishing_info / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Publishing info file not found: {yaml_path}")
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return PublishingInfo(**data)
