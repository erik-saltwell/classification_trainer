import os

import wandb
from classification_trainer.configuration.training_info import TrainingInfo


def initialize_wandb(training_info: TrainingInfo) -> wandb.Run:
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_PROJECT"] = training_info.model_name
    wandb.login()
    wandb_config = {
        "training_info": training_info.to_dict(),
    }
    return wandb.init(project=training_info.model_name, config=wandb_config)
