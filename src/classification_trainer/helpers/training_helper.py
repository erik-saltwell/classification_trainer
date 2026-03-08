from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip
from contextlib import nullcontext
from typing import Any, cast

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.integrations.integration_utils import WandbCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import TrainOutput
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from classification_trainer.configuration import DatasetInfo, TrainingInfo
from classification_trainer.configuration.chat_template_info import ChatTemplateInfo
from classification_trainer.helpers.wandb_helper import suppress_wandb_finish


class _NoFinishWandbCallback(WandbCallback):
    """WandbCallback that skips finish() — the caller owns the run lifecycle."""

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        processing_class: Any = None,
        **kwargs: Any,
    ) -> None:
        pass


def load_base_model(training_info: TrainingInfo) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=training_info.base_model_info.huggingface_name,
        max_seq_length=training_info.max_sequence_length,
        dtype=training_info.dtype,
        load_in_4bit=training_info.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    loftq_config: dict[str, int] | None = None
    if training_info.use_loftq:
        loftq_config = {"loftq_bits": training_info.loftq_bits, "loftq_iter": training_info.loftq_iter}

    model = FastLanguageModel.get_peft_model(
        model,
        training_info.sft_parameters.rank,
        target_modules=training_info.sft_parameters.training_modules,
        lora_alpha=training_info.lora_alpha,
        lora_dropout=training_info.sft_parameters.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_info.seed,
        use_rslora=training_info.use_rslora,
        loftq_config=loftq_config,
    )

    return (model, tokenizer)


def create_trainer(
    dataset_info: DatasetInfo,
    training_info: TrainingInfo,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    report_to_wandb: bool,
    eval_dataset: Dataset | None = None,
    output_dir: str | None = None,
) -> SFTTrainer:
    pre_tokenized = dataset_info.tokenized_training_column_name in train_dataset.column_names
    config: SFTConfig = training_info.create_sft_config(
        dataset_info, report_to_wandb, output_dir, pre_tokenized=pre_tokenized
    )
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
    )
    if report_to_wandb:
        trainer.remove_callback(WandbCallback)
        trainer.add_callback(_NoFinishWandbCallback())
    if training_info.train_on_outputs_only and not pre_tokenized:
        chat_template_info: ChatTemplateInfo = training_info.base_model_info.chat_template_info

        trainer = train_on_responses_only(
            trainer=trainer,
            instruction_part=chat_template_info.instruction_separator,
            response_part=chat_template_info.response_separator,
        )
    return trainer


def run_training(trainer: SFTTrainer, model: PreTrainedModel) -> int:
    """Train the model and return the final global step."""
    import wandb

    model.train()
    FastLanguageModel.for_training(model)
    ctx = suppress_wandb_finish() if wandb.run is not None else nullcontext()
    with ctx:
        training_output = cast(TrainOutput, trainer.train())
    return training_output.global_step  # pyright: ignore
