from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip
from typing import cast

from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.trainer_utils import TrainOutput
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from classification_trainer.configuration import BaseModelInfo, DatasetInfo, TrainingInfo
from classification_trainer.configuration.chat_template_info import ChatTemplateInfo


def load_base_model(
    base_model_info: BaseModelInfo, training_options: TrainingInfo
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_info.huggingface_name,
        max_seq_length=training_options.max_sequence_length,
        dtype=training_options.dtype,
        load_in_4bit=training_options.load_in_4bit,
        use_gradient_checkpointing="unsloth",
    )

    loftq_config: dict[str, int] | None = None
    if training_options.use_loftq:
        loftq_config = {"loftq_bits": training_options.loftq_bits, "loftq_iter": training_options.loftq_iter}

    model = FastLanguageModel.get_peft_model(
        model,
        training_options.sft_parameters.rank,
        target_modules=training_options.sft_parameters.training_modules,
        lora_alpha=training_options.lora_alpha,
        lora_dropout=training_options.sft_parameters.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=training_options.seed,
        use_rslora=training_options.use_rslora,
        loftq_config=loftq_config,
    )

    return (model, tokenizer)


def create_trainer(
    dataset_info: DatasetInfo,
    training_info: TrainingInfo,
    base_model_info: BaseModelInfo,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    report_to_wandb: bool,
    eval_dataset: Dataset | None = None,
) -> SFTTrainer:
    config: SFTConfig = training_info.create_sft_config(dataset_info, report_to_wandb)
    trainer: SFTTrainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
    )
    if training_info.train_on_outputs_only:
        chat_template_info: ChatTemplateInfo = base_model_info.chat_template_info

        trainer = train_on_responses_only(
            trainer=trainer,
            instruction_part=chat_template_info.instruction_separator,
            response_part=chat_template_info.response_separator,
        )
    return trainer


def run_training(
    trainer: SFTTrainer,
) -> TrainOutput:
    training_output = cast(TrainOutput, trainer.train())
    return training_output  # pyright: ignore
