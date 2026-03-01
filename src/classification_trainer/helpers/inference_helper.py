from unsloth import FastLanguageModel  # isort: skip  # Must precede all transformers imports

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from classification_trainer.protocols.logging_protocol import LoggingProtocol

import torch
from datasets import Dataset, disable_progress_bars, enable_progress_bars
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from classification_trainer.configuration.chat_template_info import ChatTemplateInfo
from classification_trainer.configuration.dataset_info import DatasetInfo
from classification_trainer.configuration.inference_info import InferenceInfo


def _clean_prompt_ending(prompt_text: str, template: ChatTemplateInfo) -> str:
    if template.assistant_newline and prompt_text.rstrip().endswith(template.response_separator.rstrip()):
        return prompt_text.rstrip() + "\n"
    return prompt_text


def _compute_eos_token_id(
    tokenizer: PreTrainedTokenizerBase,
    template: ChatTemplateInfo,
) -> int | list[int] | None:
    eos_ids = template.get_eos_token_ids(tokenizer)
    if not eos_ids:
        return None
    return eos_ids[0] if len(eos_ids) == 1 else eos_ids


def _build_generate_kwargs(
    enc: dict[str, Any],
    inference_info: InferenceInfo,
    tokenizer: PreTrainedTokenizerBase,
    eos_token_id: int | list[int] | None,
    pad_token_id: int | None,
) -> dict[str, Any]:
    gen_kwargs: dict[str, Any] = dict(
        **enc,
        max_new_tokens=inference_info.max_new_tokens,
        do_sample=inference_info.do_sample,
        temperature=inference_info.temperature,
        top_p=inference_info.top_p,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=True,
    )

    if inference_info.repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = inference_info.repetition_penalty

    return gen_kwargs


def _decode_and_trim_generated_texts(
    tokenizer: PreTrainedTokenizerBase,
    enc: dict[str, Any],
    out_ids: torch.Tensor,
    template: ChatTemplateInfo,
) -> list[str]:
    """
    Decode and trim generated tokens for a batch.

    - Decodes ONLY the newly generated portion for each row.
    - Trims using template.stop_strings.

    Assumes `out_ids` is the tensor returned by `model.generate(...)` with shape:
      [batch, prompt_len + generated_len]
    """
    prompt_len = enc["input_ids"].shape[1]

    # Slice newly generated tokens for the whole batch
    new_ids = out_ids[:, prompt_len:]

    # Decode per-row
    decoded_list = tokenizer.batch_decode(new_ids.detach().cpu(), skip_special_tokens=True)

    return [template.trim_at_stop_strings(decoded) for decoded in decoded_list]


def get_pad_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("pad_token_id is None and no usable eos_token_id found.")
    return int(pad_token_id)


@torch.inference_mode()
def generate_label_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    inference_info: InferenceInfo,
    template: ChatTemplateInfo,
) -> str:
    prompt_text = _clean_prompt_ending(prompt_text, template)

    # Unsloth's model.generate() silently resets tokenizer.padding_side to "right"
    # (https://github.com/unslothai/unsloth/issues/3283). Re-assert left-padding
    # before tokenization so subsequent batch calls are not affected.
    tokenizer.padding_side = "left"

    enc = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=template.add_special_tokens, pad_to_multiple_of=8
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    eos_token_id: int | list[int] | None = _compute_eos_token_id(tokenizer, template)

    pad_token_id = get_pad_token_id(tokenizer)
    gen_kwargs: dict[str, Any] = _build_generate_kwargs(enc, inference_info, tokenizer, eos_token_id, pad_token_id)
    out_ids = model.generate(**gen_kwargs)  # type: ignore
    return _decode_and_trim_generated_texts(tokenizer, enc, out_ids, template)[0]


def generate_label_texts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_texts: list[str],
    inference_info: InferenceInfo,
    template: ChatTemplateInfo,
) -> list[str]:
    """
    Batched inference — processes prompts one at a time to avoid Unsloth padding bugs.

    Unsloth's patched attention kernels corrupt outputs for padded (shorter) sequences
    in a batch (unsloth#267, unsloth#1456, unsloth#2939). Additionally, model.generate()
    silently resets tokenizer.padding_side to "right" (unsloth#3283). Until Unsloth fixes
    batched generation, we delegate to single-row generate_label_text for correctness.

    Args:
        prompt_texts: list of already-templated prompt strings (one per row)
    Returns:
        list[str] of decoded + trimmed completions (same length/order as prompt_texts)
    """
    return [generate_label_text(model, tokenizer, prompt, inference_info, template) for prompt in prompt_texts]


def add_inferred_column(
    dataset: Dataset,
    dataset_info: DatasetInfo,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inference_info: InferenceInfo,
    template: ChatTemplateInfo,
    batch_size: int = 32,
    load_from_cache_file: bool = False,
    logger: "LoggingProtocol | None" = None,
) -> Dataset:
    prompt_column_name = dataset_info.evaluation_instructions_column_name
    output_column_name = dataset_info.prediction_column_name

    if prompt_column_name not in dataset.column_names:
        raise KeyError(f"Prompt column '{prompt_column_name}' not found in dataset columns: {dataset.column_names}")

    if logger is None:

        def _infer_batch(batch: dict) -> dict:
            prompt_texts: list[str] = batch[prompt_column_name]
            preds: list[str] = generate_label_texts(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                inference_info=inference_info,
                template=template,
            )
            return {output_column_name: preds}

        return dataset.map(
            _infer_batch,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=load_from_cache_file,
        )

    with logger.progress("Batched inference", total=len(dataset)) as progress:

        def _infer_batch(batch: dict) -> dict:
            prompt_texts: list[str] = batch[prompt_column_name]
            preds: list[str] = generate_label_texts(
                model=model,
                tokenizer=tokenizer,
                prompt_texts=prompt_texts,
                inference_info=inference_info,
                template=template,
            )
            progress.advance(len(prompt_texts))
            return {output_column_name: preds}

        disable_progress_bars()
        try:
            return dataset.map(
                _infer_batch,
                batched=True,
                batch_size=batch_size,
                load_from_cache_file=load_from_cache_file,
            )
        finally:
            enable_progress_bars()


def setup_unsloth_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    padding_side: str = "left",
    ensure_pad_token: bool = True,
    run_unsloth_for_inference: bool = True,
    set_model_pad_token_id: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    One-time inference setup for decoder-only / Unsloth models.

    What it does:
    - model.eval()
    - (optionally) FastLanguageModel.for_inference(model)  # Unsloth speed-ups / patches
    - tokenizer.padding_side = "left" (recommended for batched generation)
    - ensures tokenizer.pad_token is set (defaults to eos_token)
    - optionally propagates pad_token_id into model.config / model.generation_config

    Call this ONCE after loading the model+tokenizer.
    """
    model.eval()

    # Apply Unsloth inference patch once (safe no-op if already applied)
    if run_unsloth_for_inference:
        patched = FastLanguageModel.for_inference(model)  # may return model or None
        if patched is not None:
            model = patched

    # Tokenizer padding behavior for batched decoder-only generation
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got: {padding_side!r}")
    tokenizer.padding_side = padding_side

    # Ensure a pad token exists (HF generate() wants pad_token_id for batched inputs)
    if ensure_pad_token and tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has no pad_token and no eos_token; cannot auto-assign pad_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Keep model configs consistent with tokenizer (helps avoid warnings / edge cases)
    if set_model_pad_token_id:
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            if getattr(model.config, "pad_token_id", None) is None:
                model.config.pad_token_id = pad_id  # pyright: ignore
            gen_cfg = getattr(model, "generation_config", None)
            if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
                gen_cfg.pad_token_id = pad_id

    return model, tokenizer
