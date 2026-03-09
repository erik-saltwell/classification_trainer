import os
from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum

import wandb
from classification_trainer.configuration.dataset_info import DatasetInfo
from classification_trainer.configuration.training_info import TrainingInfo


class WandBJobType(StrEnum):
    """Internal slugs used for format directories and HuggingFace repo suffixes.

    All formats — including GGUF — use a single slug. Multiple GGUF quantization
    levels are stored as separate files (``<model>-gguf-<quant>.gguf``) inside one
    shared ``gguf/`` directory and published to one shared HuggingFace repository.
    """

    TRAINING = "training"
    SWEEP = "sweep"
    BATCH_COMPUTE = "compute_batch"
    NONE = "none"


@contextmanager
def suppress_wandb_finish() -> Generator[None, None, None]:
    """Suppress wandb.finish() calls during training.

    Unsloth's compiled trainer wrapper unconditionally calls wandb.finish()
    after trainer.train() to support multi-run notebook scenarios. This
    context manager suppresses that call so the caller retains ownership
    of the run lifecycle.
    """
    _original = wandb.finish
    wandb.finish = lambda *args, **kwargs: None
    try:
        yield
    finally:
        wandb.finish = _original


def get_base_model_full_from_training_info(training_info: TrainingInfo) -> str:
    return training_info.base_model


def get_base_model_slug_from_training_info(training_info: TrainingInfo) -> str:
    full_name: str = get_base_model_full_from_training_info(training_info)
    parts = full_name.split("-", 1)
    result = parts[0] if len(parts) > 0 else "no-model-name"
    return result


def get_simple_dataset_name_from_dataset_info(dataset_info: DatasetInfo) -> str:
    parts = dataset_info.huggingface_name.split("/", 1)
    result = parts[1] if len(parts) > 1 else dataset_info.huggingface_name
    return result


def get_tags(training_info: TrainingInfo, dataset_info: DatasetInfo) -> list[str]:
    tags: list[str] = [
        get_base_model_full_from_training_info(training_info),
        get_base_model_slug_from_training_info(training_info),
        get_simple_dataset_name_from_dataset_info(dataset_info),
    ]
    if training_info.has_sweep:
        tags.append("sweep")

    return tags


def initialize_wandb(training_info: TrainingInfo, dataset_info: DatasetInfo, job_type: WandBJobType) -> wandb.Run:
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    group_name = "manual_run"
    if training_info.sweep_config is not None:
        group_name = "swp_" + training_info.sweep_config.sweep_name

    wandb.login()
    return wandb.init(
        project=training_info.wandb_project_name,
        group=group_name,
        job_type=str(job_type),
        tags=get_tags(training_info, dataset_info),
        config={"training_info": training_info.to_dict()},
    )
