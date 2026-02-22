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
