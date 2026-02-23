from __future__ import annotations

from unsloth import FastLanguageModel  # isort: skip
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from classification_trainer.configuration import BaseModelInfo, TrainingInfo


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
