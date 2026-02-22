import unsloth  # isort: skip
from typing import cast

from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import BaseModelInfo


def load_tokenizer_from_hf(base_model_info: BaseModelInfo) -> PreTrainedTokenizerBase:
    """Load and return the tokenizer for the given base model name."""

    tokenizer = cast(
        PreTrainedTokenizerBase, unsloth.tokenizer_utils.load_correct_tokenizer(base_model_info.huggingface_name)
    )
    return tokenizer


def generate_eos(tokenizer: PreTrainedTokenizerBase) -> str:
    eos = tokenizer.eos_token
    if eos is None:
        raise ValueError("The tokenizer does not have an EOS token defined.")
    if not isinstance(eos, str):
        raise ValueError(f"Expected eos_token to be a str, got {type(eos).__name__}")
    return eos


def apply_chat_template(
    instruction: str, user_input: str, output: str, tokenizer: PreTrainedTokenizerBase, eos: str
) -> str:
    return_value: str
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]

    if output:
        messages.append({"role": "assistant", "content": output})
        return_value = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        # Some tokenizers omit the EOS token at the end of the assistant turn; append it
        # explicitly so the model learns to terminate its own outputs during training.
        if not return_value.rstrip().endswith(eos):
            return_value = return_value.rstrip() + eos
    else:
        return_value = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    return return_value
