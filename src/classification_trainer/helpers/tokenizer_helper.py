from __future__ import annotations

import unsloth  # isort: skip

from typing import cast

from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import BaseModelInfo


def load_tokenizer_from_hf(base_model_info: BaseModelInfo) -> PreTrainedTokenizerBase:
    """Load and return the tokenizer for the given base model name."""
    if unsloth is not None:
        from transformers import PreTrainedTokenizerBase

        tokenizer = cast(
            PreTrainedTokenizerBase, unsloth.tokenizer_utils.load_correct_tokenizer(base_model_info.huggingface_name)
        )
        return tokenizer

    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(base_model_info.huggingface_name))
    return tokenizer
