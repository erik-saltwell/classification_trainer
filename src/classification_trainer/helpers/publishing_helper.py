from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, ModelCard
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from classification_trainer.configuration import (
    BaseModelInfo,
    ChatTemplateInfo,
    DatasetInfo,
    PublishingInfo,
    TrainingInfo,
)
from classification_trainer.configuration.publishing_info import SaveFormat
from classification_trainer.configuration.training_info import TrainingLengthType
from classification_trainer.protocols import LoggingProtocol
from classification_trainer.protocols.metric_result import MetricResult
from classification_trainer.utils.common_paths import CommonPaths
from classification_trainer.utils.flush_gpu_memory import flush_gpu_memory

# ---------------------------------------------------------------------------
# Format-specific private save functions
# ---------------------------------------------------------------------------


def _save_lora(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, save_dir: Path) -> None:
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))


def _save_gguf_quant(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    quantization_method: str,
    output_filename: str,
) -> None:
    """Save one GGUF quantization file to ``save_dir`` with a controlled filename.

    Unsloth uses the first argument as a temp dir for the HF-format merge step,
    then writes the final .gguf file(s) to the current working directory.
    We snapshot CWD before the call and move any new .gguf files into ``save_dir``
    under ``output_filename`` after the call.
    If the model is sharded into multiple .gguf files, each shard is moved
    with an index suffix appended to ``output_filename``.
    """
    cwd = Path.cwd()
    existing_gguf = set(cwd.glob("*.gguf"))
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained_gguf(tmp_dir, tokenizer, quantization_method=quantization_method)
    gguf_files = sorted(set(cwd.glob("*.gguf")) - existing_gguf)
    if not gguf_files:
        raise RuntimeError(f"No .gguf file was created for quantization '{quantization_method}'")
    if len(gguf_files) == 1:
        shutil.move(str(gguf_files[0]), str(save_dir / output_filename))
    else:
        # Multi-shard model: append shard index to stem
        stem = output_filename.removesuffix(".gguf")
        total = len(gguf_files)
        for idx, shard in enumerate(gguf_files, start=1):
            dest = save_dir / f"{stem}-{idx:05d}-of-{total:05d}.gguf"
            shutil.move(str(shard), str(dest))


def _save_merged(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    save_dir: Path,
    save_method: str,
) -> None:
    model.save_pretrained_merged(str(save_dir), tokenizer, save_method=save_method)


# ---------------------------------------------------------------------------
# Modelfile generation (Ollama / llama.cpp)
# ---------------------------------------------------------------------------


def _build_template_body(chat_template_info: ChatTemplateInfo) -> str:
    end_of_turn = chat_template_info.stop_strings[0] if chat_template_info.stop_strings else ""
    instr_sep = chat_template_info.instruction_separator
    resp_sep = chat_template_info.response_separator
    sys_sep = chat_template_info.system_separator

    if sys_sep is not None:
        system_part = f"{{{{- if .System }}}}{sys_sep}{{{{ .System }}}}{end_of_turn}\n{{{{- end }}}}\n"
    else:
        system_part = "{{- if .System }}{{ .System }}\n{{- end }}\n"

    return system_part + f"{{{{- if .Prompt }}}}{instr_sep}{{{{ .Prompt }}}}{end_of_turn}\n{{{{- end }}}}\n" + resp_sep


def generate_modelfile(
    save_dir: Path,
    format_slug: str,
    training_info: TrainingInfo,
    publishing_info: PublishingInfo,
) -> None:
    """Generate an Ollama-compatible Modelfile in ``save_dir``.

    Supported formats: GGUF (FROM = relative .gguf filename) and merged
    (FROM = HuggingFace repo ID).  LoRA is not supported and must not be
    passed as ``format_slug``.
    """
    chat_template_info = training_info.base_model_info.chat_template_info
    inference_info = training_info.inference_info

    # --- FROM line ---
    if format_slug == SaveFormat.GGUF:
        quant = publishing_info.gguf_quantizations[0]
        from_line = f"FROM {training_info.model_name}-gguf-{quant}.gguf"
    else:  # MERGED
        from_line = f"FROM {training_info.hugging_face_user_name}/{training_info.model_name}-merged"

    # --- SYSTEM block ---
    system_block = f'SYSTEM """\n{training_info.system_prompt}\n"""'

    # --- TEMPLATE block ---
    template_body = _build_template_body(chat_template_info)
    template_block = f'TEMPLATE """\n{template_body}"""'

    # --- PARAMETER lines ---
    param_lines: list[str] = [
        f"PARAMETER temperature {inference_info.temperature}",
        f"PARAMETER top_p {inference_info.top_p}",
        f"PARAMETER num_predict {inference_info.max_new_tokens}",
        f"PARAMETER num_ctx {training_info.max_sequence_length}",
    ]
    if inference_info.repetition_penalty is not None:
        param_lines.append(f"PARAMETER repeat_penalty {inference_info.repetition_penalty}")
    for stop in chat_template_info.stop_strings:
        param_lines.append(f'PARAMETER stop "{stop}"')

    content = "\n\n".join([from_line, system_block, template_block, "\n".join(param_lines)])
    (save_dir / "Modelfile").write_text(content, encoding="utf-8")


def generate_gguf_hf_metadata(
    save_dir: Path,
    training_info: TrainingInfo,
    publishing_info: PublishingInfo,
) -> None:
    """Write HuggingFace Ollama metadata files (template, system, params) to ``save_dir``.

    These files enable ``ollama run hf.co/<user>/<repo>`` without any local download.
    Called only for SaveFormat.GGUF.
    """
    chat_template_info = training_info.base_model_info.chat_template_info
    inference_info = training_info.inference_info

    (save_dir / "template").write_text(_build_template_body(chat_template_info), encoding="utf-8")
    (save_dir / "system").write_text(training_info.system_prompt, encoding="utf-8")

    params: dict = {
        "temperature": inference_info.temperature,
        "top_p": inference_info.top_p,
        "num_predict": inference_info.max_new_tokens,
        "num_ctx": training_info.max_sequence_length,
        "stop": list(chat_template_info.stop_strings),
    }
    if inference_info.repetition_penalty is not None:
        params["repeat_penalty"] = inference_info.repetition_penalty
    (save_dir / "params").write_text(json.dumps(params, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Model card generation
# ---------------------------------------------------------------------------


def _metrics_table(metrics: list[MetricResult]) -> str:
    rows = "\n".join(f"| {m.metric_name} | {m.metric_result} |" for m in metrics)
    return f"| Metric | Value |\n|--------|-------|\n{rows}"


def _usage_section(
    format_slug: str,
    repo_id: str,
    base_model_name: str,
    gguf_quantizations: list[str] | None = None,
) -> str:
    if format_slug == SaveFormat.GGUF:
        model_repo_name = repo_id.split("/")[-1]  # e.g. "my-classifier-gguf"
        file_lines = "\n".join(f"- `{model_repo_name}-{quant}.gguf`" for quant in (gguf_quantizations or []))
        return (
            "### GGUF with llama.cpp / Ollama\n\n"
            f"Repository: `{repo_id}`\n\n"
            "Available quantization files:\n\n"
            f"{file_lines}\n\n"
            "#### Load with llama.cpp\n\n"
            "```bash\n"
            f"llama-cli -m {model_repo_name}-q8_0.gguf\n"
            "```"
        )
    if format_slug == SaveFormat.LORA:
        return (
            "### LoRA with Python (PEFT)\n\n"
            "```python\n"
            "from peft import PeftModel\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
            f'base_model = AutoModelForCausalLM.from_pretrained("{base_model_name}")\n'
            f'model = PeftModel.from_pretrained(base_model, "{repo_id}")\n'
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
            "```"
        )
    if format_slug == SaveFormat.MERGED:
        return (
            "### Merged Checkpoint\n\n"
            "```python\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
            f'model = AutoModelForCausalLM.from_pretrained("{repo_id}")\n'
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
            "```"
        )
    return ""


def generate_model_card(
    save_dir: Path,
    format_slug: str,
    training_info: TrainingInfo,
    dataset_info: DatasetInfo,
    base_model_info: BaseModelInfo,
    publishing_info: PublishingInfo,
    pre_metrics: list[MetricResult],
    post_metrics: list[MetricResult],
) -> None:
    model_name = training_info.model_name
    username = training_info.hugging_face_user_name
    repo_id = f"{username}/{model_name}-{format_slug}"

    # Model details
    model_detail_lines = [
        f"- **Base model**: [{base_model_info.huggingface_name}]"
        f"(https://huggingface.co/{base_model_info.huggingface_name})",
        f"- **LoRA rank**: {training_info.sft_parameters.rank}",
    ]
    if format_slug == SaveFormat.GGUF:
        quants_str = ", ".join(f"`{q}`" for q in publishing_info.gguf_quantizations)
        model_detail_lines.append(f"- **Quantizations**: {quants_str}")
    model_details = "\n".join(model_detail_lines)

    # Dataset
    split_lines = [
        f"- **Dataset**: [{dataset_info.huggingface_name}]"
        f"(https://huggingface.co/datasets/{dataset_info.huggingface_name})",
        f"- **Training split**: `{dataset_info.training_split_name}`",
    ]
    if dataset_info.test_split_name:
        split_lines.append(f"- **Test split**: `{dataset_info.test_split_name}`")
    if dataset_info.validation_split_name:
        split_lines.append(f"- **Validation split**: `{dataset_info.validation_split_name}`")
    split_lines.append(f"- **Positive class**: `{dataset_info.positive_case}`")
    dataset_section = "\n".join(split_lines)

    # Training configuration
    if training_info.training_length_type == TrainingLengthType.STEPS:
        length_line = f"- **Steps**: {int(training_info.training_length)}"
    else:
        length_line = f"- **Epochs**: {training_info.training_length}"
    training_config = "\n".join(
        [
            length_line,
            f"- **Batch size (per device)**: {training_info.per_device_batch_size}",
            f"- **Gradient accumulation steps**: {training_info.gradient_accumulation_steps}",
            f"- **Learning rate**: {training_info.sft_parameters.learning_rate}",
            f"- **Max sequence length**: {training_info.max_sequence_length}",
        ]
    )

    # Pre-training metrics
    if pre_metrics:
        pre_section = f"## Pre-Training Metrics\n\n{_metrics_table(pre_metrics)}"
    else:
        pre_section = "## Pre-Training Metrics\n\n_Pre-training evaluation was not run._"

    content = "\n".join(
        [
            f"# {model_name} ({format_slug})",
            "",
            training_info.model_card_description,
            "",
            "## Model Details",
            "",
            model_details,
            "",
            "## Dataset",
            "",
            dataset_section,
            "",
            "## Training Configuration",
            "",
            training_config,
            "",
            pre_section,
            "",
            f"## Post-Training Metrics\n\n{_metrics_table(post_metrics)}",
            "",
            "## Usage",
            "",
            _usage_section(
                format_slug,
                repo_id,
                base_model_info.huggingface_name,
                gguf_quantizations=publishing_info.gguf_quantizations,
            ),
        ]
    )

    ModelCard(content).save(save_dir / "README.md")


# ---------------------------------------------------------------------------
# save_model orchestrator
# ---------------------------------------------------------------------------


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    training_info: TrainingInfo,
    dataset_info: DatasetInfo,
    pre_metrics: list[MetricResult],
    post_metrics: list[MetricResult],
    logger: LoggingProtocol,
) -> None:
    if training_info.publishing_info is None:
        raise ValueError("No publishing Info found in training info")

    publishing_info = training_info.publishing_info

    if not publishing_info.any_save_enabled:
        return

    logger.report_message("[blue]Saving model artifacts...[/blue]")

    def _save_format(slug: str) -> None:
        save_dir = CommonPaths.get().get_model_save_directory(training_info.model_name, slug)
        save_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger.report_message(f"Saving {slug} \u2192 {save_dir}/")
            if slug == SaveFormat.LORA:
                _save_lora(model, tokenizer, save_dir)
            elif slug == SaveFormat.MERGED:
                _save_merged(model, tokenizer, save_dir, publishing_info.merged_save_method)
            elif slug == SaveFormat.GGUF:
                # All quantizations share one directory; each quant is a separate file
                repo_name = f"{training_info.model_name}-{SaveFormat.GGUF}"
                for quant in publishing_info.gguf_quantizations:
                    filename = f"{repo_name}-{quant}.gguf"
                    logger.report_message(f"    Quantizing {quant} \u2192 {filename}")
                    _save_gguf_quant(model, tokenizer, save_dir, quant, filename)
            generate_model_card(
                save_dir,
                slug,
                training_info,
                dataset_info,
                training_info.base_model_info,
                publishing_info,
                pre_metrics,
                post_metrics,
            )
            if slug in (SaveFormat.GGUF, SaveFormat.MERGED):
                logger.report_message(f"    Generating Modelfile \u2192 {save_dir}/Modelfile")
                generate_modelfile(save_dir, slug, training_info, publishing_info)
            # HF Ollama metadata files are GGUF-only; not applicable to merged or lora
            if slug == SaveFormat.GGUF:
                logger.report_message(f"    Generating HF metadata \u2192 {save_dir}/{{template,system,params}}")
                generate_gguf_hf_metadata(save_dir, training_info, publishing_info)
            logger.report_message(f"  \u2713 {slug}")
            flush_gpu_memory()
        except Exception:
            shutil.rmtree(save_dir, ignore_errors=True)
            raise

    if publishing_info.save_lora:
        _save_format(SaveFormat.LORA)
    if publishing_info.save_gguf:
        _save_format(SaveFormat.GGUF)
    if publishing_info.save_merged:
        _save_format(SaveFormat.MERGED)


# ---------------------------------------------------------------------------
# publish_model orchestrator
# ---------------------------------------------------------------------------


def publish_model(
    training_info: TrainingInfo,
    publishing_info: PublishingInfo,
    logger: LoggingProtocol,
) -> None:
    if not publishing_info.any_publish_enabled:
        return

    logger.report_message("[blue]Publishing model artifacts to HuggingFace Hub...[/blue]")

    slugs: list[str] = []
    if publishing_info.publish_lora:
        slugs.append(SaveFormat.LORA)
    if publishing_info.publish_gguf:
        slugs.append(SaveFormat.GGUF)
    if publishing_info.publish_merged:
        slugs.append(SaveFormat.MERGED)

    failures: list[str] = []
    api = HfApi()

    for slug in slugs:
        save_dir = CommonPaths.get().get_model_save_directory(training_info.model_name, slug)
        repo_id = f"{training_info.hugging_face_user_name}/{training_info.model_name}-{slug}"

        if not save_dir.exists():
            msg = (
                f"[red]Error: No saved artifacts found at {save_dir}.\n"
                f"Run `train --publishing-info <name>` first.[/red]"
            )
            logger.report_message(msg)
            failures.append(slug)
            continue

        readme = save_dir / "README.md"
        if not readme.exists():
            msg = (
                f"[red]Error: Missing model card at {readme}.\n"
                f"Re-run `train --publishing-info <name>` to regenerate.[/red]"
            )
            logger.report_message(msg)
            failures.append(slug)
            continue

        try:
            if slug in (SaveFormat.GGUF, SaveFormat.MERGED) and not (save_dir / "Modelfile").exists():
                logger.report_message(f"    Generating missing Modelfile \u2192 {save_dir}/Modelfile")
                generate_modelfile(save_dir, slug, training_info, publishing_info)
            if slug == SaveFormat.GGUF and not all((save_dir / f).exists() for f in ("template", "system", "params")):
                logger.report_message(f"    Generating missing HF metadata \u2192 {save_dir}/")
                generate_gguf_hf_metadata(save_dir, training_info, publishing_info)
            logger.report_message(f"Publishing {slug} \u2192 {repo_id}")
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=publishing_info.private)
            api.upload_folder(folder_path=str(save_dir), repo_id=repo_id, repo_type="model")
            logger.report_message(f"  \u2713 {slug}")
        except Exception as exc:
            logger.report_message(f"[red]Error publishing {slug}: {exc}[/red]")
            failures.append(slug)

    if failures:
        failed = ", ".join(failures)
        raise RuntimeError(f"Publishing failed for format(s): {failed}")
