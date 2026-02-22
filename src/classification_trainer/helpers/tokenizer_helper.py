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


def _generate_eos(tokenizer: PreTrainedTokenizerBase) -> str:
    eos = tokenizer.eos_token
    if eos is None:
        raise ValueError("The tokenizer does not have an EOS token defined.")
    if isinstance(eos, list):
        eos = eos[0]
    eos_token: str = eos
    return eos_token


def _apply_chat_template(
    instruction: str, input: str, output: str, tokenizer: PreTrainedTokenizerBase, eos: str | None
) -> str:
    return_value: str
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input},
    ]

    if output:
        messages.append({"role": "assistant", "content": output})
        return_value = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        if eos and not return_value.rstrip().endswith(eos):
            return_value = return_value.rstrip() + eos
    else:
        return_value = str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    return return_value
