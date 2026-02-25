from __future__ import annotations

from typing import Any, cast

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from classification_trainer.configuration import DatasetInfo


def run_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    dataset_info: DatasetInfo,
    batch_size: int = 4,
) -> list[str]:
    """Generate raw model outputs for each row in the dataset.

    Args:
        model: A model in inference mode (FastLanguageModel.for_inference already called).
        tokenizer: The tokenizer matching the model.
        dataset: Dataset that already contains the evaluation_instructions_column_name column.
        dataset_info: Provides column name lookups.
        batch_size: Number of prompts to process per generate() call.

    Returns:
        List of decoded response strings (new tokens only, prompt excluded) for each row.
    """
    prompts: list[str] = dataset[dataset_info.evaluation_instructions_column_name]
    results: list[str] = []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            encoding = tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
            inputs = {k: v.to(model.device) for k, v in encoding.items()}

            input_length = cast(torch.Tensor, encoding["input_ids"]).shape[1]

            with torch.no_grad():
                # cast to Any: generate() is set dynamically on PreTrainedModel instances,
                # so pyright falls back to nn.Module.__getattr__ -> Tensor | Module.
                output_ids: torch.Tensor = cast(Any, model).generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (not the prompt) so that
            # ResponseCleaner receives just the model response. Decoding the full
            # sequence with skip_special_tokens=True strips <|im_start|> from the
            # decoded text, making the response-separator split in ResponseCleaner
            # fail and applying cleaning rules against the whole conversation instead.
            decoded = tokenizer.batch_decode(output_ids[:, input_length:], skip_special_tokens=True)
            results.extend(decoded)
    finally:
        tokenizer.padding_side = original_padding_side

    return results
