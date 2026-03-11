from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip
import json
from contextlib import nullcontext
from typing import Any, cast

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.integrations.integration_utils import WandbCallback
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import TrainOutput
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from classification_trainer.configuration import DatasetInfo, TrainingInfo
from classification_trainer.configuration.chat_template_info import ChatTemplateInfo
from classification_trainer.helpers.wandb_helper import suppress_wandb_finish
from classification_trainer.protocols.logging_protocol import LoggingProtocol, ProgressTask


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


class _ProgressLoggingCallback(TrainerCallback):
    """Bridges HuggingFace Trainer events to LoggingProtocol for progress reporting."""

    _EXCLUDED_LOG_KEYS = frozenset(
        {
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "total_floss",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
        }
    )

    def __init__(self, logger: LoggingProtocol, progress_task: ProgressTask) -> None:
        self._logger = logger
        self._progress = progress_task

    def on_train_begin(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        self._progress.set_total(state.max_steps)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        self._progress.set_completed(state.global_step)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs and self._logger.verbose_training:
            filtered = {k: v for k, v in logs.items() if k not in self._EXCLUDED_LOG_KEYS}
            if filtered:
                self._logger.report_message(json.dumps(filtered))


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
    if training_info.sft_parameters.use_loftq:
        loftq_config = {
            "loftq_bits": training_info.sft_parameters.loftq_bits,
            "loftq_iter": training_info.sft_parameters.loftq_iter,
        }

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


def run_training(
    trainer: SFTTrainer,
    model: PreTrainedModel,
    logger: LoggingProtocol | None = None,
) -> int:
    """Train the model and return the final global step."""
    import wandb

    model.train()
    FastLanguageModel.for_training(model)
    ctx = suppress_wandb_finish() if wandb.run is not None else nullcontext()
    progress_ctx = logger.progress("Training") if logger is not None else nullcontext()

    with ctx, progress_ctx as progress_task:
        if progress_task is not None:
            trainer.add_callback(_ProgressLoggingCallback(logger, progress_task))  # type: ignore[arg-type]
        training_output = cast(TrainOutput, trainer.train())
    return training_output.global_step  # pyright: ignore
