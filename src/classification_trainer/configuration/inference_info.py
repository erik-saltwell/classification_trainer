import yaml
from pydantic import BaseModel, ConfigDict

from classification_trainer.utils.common_paths import CommonPaths


class InferenceInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    # Deterministic decoding for stable labels
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0

    # Keep completions short (pos/neg)
    max_new_tokens: int = 8

    # Optional: discourage rambling if your prompt isn't strict enough
    repetition_penalty: float | None = None

    # If you're using Unsloth, flip fast inference mode before generate
    prepare_unsloth_inference: bool = True
    padding_side: str = "left"
    ensure_pad_token: bool = True
    set_model_pad_token_id: bool = True
    metrics: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "total_seen")
    sweep_metric: str = "f1"
    sweep_metric_goal: str = "maximize"


def load_inference_info(name: str) -> InferenceInfo:
    """Load a DatasetInfo from a YAML file in the dataset_info directory.

    Args:
        name: The yaml filename without extension (e.g. "reddit-rpg").

    Returns:
        A validated DatasetInfo instance.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
        pydantic.ValidationError: If the file data fails validation.
    """
    yaml_path = CommonPaths.get().inference_info / f"{name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset info file not found: {yaml_path}")
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    return InferenceInfo(**data)
