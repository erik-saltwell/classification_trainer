from __future__ import annotations

from enum import StrEnum

import yaml
from pydantic import BaseModel, Field
from trl.trainer.sft_config import SFTConfig

from classification_trainer.utils.text_fragments import FragmentID, get_fragment

from .base_model_info import BaseModelInfo, load_base_model_info
from .dataset_info import DatasetInfo
from .inference_info import InferenceInfo, load_inference_info
from .publishing_info import PublishingInfo, load_publishing_info
from .sft_parameters import SFTParameters
from .sweep_config import SweepConfig
from .wandb_config import WandbConfig


class TrainingLengthType(StrEnum):
    STEPS = "steps"
    EPOCHS = "epoch"


_NO: str = "no"


class TrainingInfo(BaseModel):
    model_name: str  # must be valid hugging face name, use hf_validators.py
    hugging_face_user_name: str  # must be valid hugging face name, use hf_validators.py
    system_prompt_name: FragmentID
    # References to reusable config files (filename stem, no .yaml extension)
    base_model: str
    inference: str
    publishing: str | None = None
    model_card_description: str
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
    use_loftq: bool = False
    loftq_bits: int = 4
    loftq_iter: int = 1
    sft_parameters: SFTParameters = Field(default_factory=SFTParameters)
    rslora: bool = False
    evaluation_steps: int = 50
    evaluation_enabled: bool = True
    greater_is_better: bool = False
    wandb_config: WandbConfig | None = None
    sweep: SweepConfig | None = None

    @property
    def base_model_info(self) -> BaseModelInfo:
        return load_base_model_info(self.base_model)

    @property
    def inference_info(self) -> InferenceInfo:
        return load_inference_info(self.inference)

    @property
    def publishing_info(self) -> PublishingInfo | None:
        if self.publishing is None:
            return None

        return load_publishing_info(self.publishing)

    @property
    def load_best_model_at_end(self) -> bool:
        return self.evaluation_enabled

    @property
    def lora_alpha(self) -> int:
        return self.sft_parameters.rank * self.sft_parameters.alpha_multiplier

    @property
    def use_rslora(self) -> bool:
        return self.rslora or self.sft_parameters.rank > 16

    @property
    def hugging_face_model_name(self) -> str:
        return self.hugging_face_user_name + "/" + self.model_name

    @property
    def system_prompt(self) -> str:
        return get_fragment(self.system_prompt_name)

    @property
    def evaluation_strategy(self) -> str:
        return TrainingLengthType.STEPS if self.evaluation_enabled else _NO

    @property
    def save_strategy(self) -> str:
        return TrainingLengthType.STEPS if self.evaluation_enabled else _NO

    @property
    def metric_for_best_model(self) -> str | None:
        return "eval_loss" if self.evaluation_enabled else None

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict with all enum values serialized to strings."""
        return self.model_dump(mode="json")

    def create_sft_config(
        self,
        dataset_info: DatasetInfo,
        report_to_wandb: bool,
        output_dir: str | None = None,
        pre_tokenized: bool = False,
    ) -> SFTConfig:
        max_steps: int = -1 if self.training_length_type == TrainingLengthType.EPOCHS else round(self.training_length)
        max_epochs: float = -1.0 if self.training_length_type == TrainingLengthType.STEPS else self.training_length

        return SFTConfig(
            dataset_text_field=dataset_info.training_column_name,
            max_length=self.max_sequence_length,
            packing=self.packing,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_ratio=self.sft_parameters.warmup_ratio,
            max_steps=max_steps,
            num_train_epochs=max_epochs,
            learning_rate=self.sft_parameters.learning_rate,
            optim=self.sft_parameters.optim,
            weight_decay=self.sft_parameters.weight_decay,
            lr_scheduler_type=self.sft_parameters.lr_scheduler_type,
            output_dir=output_dir if output_dir is not None else self.model_name,
            seed=self.seed,
            logging_steps=10,
            save_steps=self.evaluation_steps,
            save_strategy=self.save_strategy,
            eval_strategy=self.evaluation_strategy,
            eval_steps=self.evaluation_steps,
            fp16_full_eval=True,  # makes eval cheaper on VRAM
            report_to="wandb" if report_to_wandb else "none",  # enable logging to W&B
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
        )


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
