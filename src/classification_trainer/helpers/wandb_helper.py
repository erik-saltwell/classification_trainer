import os
from collections.abc import Generator
from contextlib import contextmanager

import wandb
from classification_trainer.configuration.training_info import TrainingInfo


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


def initialize_wandb(training_info: TrainingInfo) -> wandb.Run:
    assert training_info.wandb_config is not None
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    cfg = training_info.wandb_config
    wandb.login()
    return wandb.init(
        project=cfg.project,
        group=cfg.group,
        job_type=cfg.job_type,
        tags=[cfg.sweep_name] if cfg.sweep_name else [],
        config={"training_info": training_info.to_dict()},
    )
