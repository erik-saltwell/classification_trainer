# Research: Model Save and Publish

**Feature**: 001-model-save-publish
**Date**: 2026-03-05

---

## 1. Unsloth GGUF Save API

**Decision**: Use `model.save_pretrained_gguf(output_dir, tokenizer, quantization_method=...)`.

**Rationale**: Unsloth exposes a first-class GGUF save method directly on the patched model
object. It handles llama.cpp conversion internally, requires no additional CLI tooling, and
accepts a `quantization_method` string (e.g., `"q8_0"`, `"q4_k_m"`, `"f16"`).

**Alternatives considered**:
- `llama.cpp` convert scripts invoked via subprocess — requires llama.cpp installed separately,
  fragile path management, no benefit over the built-in Unsloth path.

---

## 2. Unsloth LoRA Adapter Save API

**Decision**: Use `model.save_pretrained(output_dir)` + `tokenizer.save_pretrained(output_dir)`.

**Rationale**: Unsloth models are PeftModel subclasses. Calling `save_pretrained` on a LoRA-wrapped
Unsloth model saves only the LoRA adapter weights and config (not the full base model), which is
the correct behavior for a LoRA artifact. This is idiomatic PEFT/HuggingFace and requires no
additional dependencies.

**Alternatives considered**:
- Manually extracting `model.peft_config` and state dict — unnecessary; `save_pretrained` handles
  all of this correctly.

---

## 3. Unsloth Merged Checkpoint Save API

**Decision**: Use `model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")`.

**Rationale**: Unsloth provides a first-class merged-save method that fuses LoRA adapter weights
into the base model and saves a standard HuggingFace checkpoint. `merged_16bit` is the safest
default (full precision), suitable for vLLM and further quantization. Memory cost is approximately
2x the base model size in VRAM.

**Alternatives considered**:
- `model.merge_and_unload()` followed by standard `save_pretrained` — equivalent result but
  Unsloth's method includes additional safety checks and quantization options.
- `merged_4bit_forced` — lower memory, less precision; can be offered as a per-format option
  in `PublishingInfo` if needed later. Default to 16-bit.

---

## 4. AWQ Save

**Decision**: Use the `autoawq` library (`pip install autoawq`) to quantize a saved merged
checkpoint. Process: (a) save merged checkpoint to a temp path, (b) load with
`AutoAWQForCausalLM.from_pretrained`, (c) quantize with `model.quantize(tokenizer, quant_config)`,
(d) save with `model.save_quantized(output_dir)` + `tokenizer.save_pretrained(output_dir)`.

**Rationale**: AutoAWQ is the standard library for AWQ quantization of HuggingFace-compatible
models. It supports all common LLM architectures (Llama, Mistral, Qwen, etc.) and produces
vLLM-compatible artifacts.

**Constraints**:
- Requires a calibration dataset (default: `pileval` from AutoAWQ's built-in defaults — no
  user config needed for standard quantization).
- Requires a GPU with sufficient VRAM (roughly 2x base model size). Will fail with an OOM error
  if insufficient; this should be caught and reported as an actionable error.
- `autoawq` is an optional dependency; it should be installed only when AWQ save is enabled.
  Recommend adding as an optional extra in `pyproject.toml` (e.g., `pip install classification-trainer[awq]`).

**Alternatives considered**:
- GPTQ quantization — older format; AWQ is now preferred for vLLM compatibility and higher
  quality at the same bit-width.
- Unsloth AWQ path — Unsloth does not natively support AWQ save; only GGUF and merged saves.

---

## 5. HuggingFace Hub Upload

**Decision**: Use `huggingface_hub.HfApi`:
```python
api = HfApi()
api.create_repo(repo_id="user/model-gguf", repo_type="model", exist_ok=True, private=True)
api.upload_folder(folder_path=str(save_dir), repo_id="user/model-gguf", repo_type="model")
```

**Rationale**: `huggingface_hub` is already a transitive dependency (via `transformers`). Using
`upload_folder` uploads all files in a directory atomically, including the `README.md` model
card. `create_repo(..., exist_ok=True)` is idempotent. Repositories are created private by
default.

**Alternatives considered**:
- `model.push_to_hub(...)` — Unsloth's built-in push method; does not use the saved
  model card on disk (regenerates its own card), which violates FR-012.
- Manual file-by-file upload via `api.upload_file` — more control but much more complex;
  `upload_folder` is simpler and correct.

---

## 6. Model Card Generation

**Decision**: Generate a plain Markdown `README.md` file using Python f-strings / string
templates. Use `huggingface_hub.ModelCard` to write the final file.

**Model card sections**:
1. Title (model name + format)
2. Description (from `PublishingInfo.description`)
3. Model Details: base model name, LoRA rank, quantization
4. Dataset: HuggingFace dataset name, split info, positive class
5. Training Configuration: epochs/steps, batch size, learning rate, max sequence length
6. Evaluation Metrics: pre-training and post-training classification metrics (from `list[MetricResult]`)
7. Usage: format-specific instructions (Ollama `ollama run`, Python vLLM snippet, PEFT load snippet, HF pipeline snippet)

**Rationale**: `ModelCard` from `huggingface_hub` handles writing the Markdown with
correct HuggingFace metadata frontmatter (YAML header). Using the library avoids
formatting mistakes and ensures Hub compatibility.

**Alternatives considered**:
- LLM-generated prose — not deterministic, adds latency, unnecessary.
- Jinja2 templates — adds a dependency; f-strings are sufficient for this use case.

---

## 7. PublishingInfo Config Schema

**Decision**: New Pydantic model `PublishingInfo` in `configuration/publishing_info.py`, loaded
from `publishing_info/<name>.yaml`. Fields:

```yaml
description: "Description shown in model cards"
gguf_quantization: q8_0           # optional, default: q8_0
merged_save_method: merged_16bit  # optional, default: merged_16bit
save_formats:                     # list of: gguf, lora, merged, awq
  - lora
  - gguf
publish_formats:                  # list of: gguf, lora, merged, awq
  - gguf
```

**Rationale**: Consistent with `TrainingInfo`, `DatasetInfo` patterns. Validates format
names against the `SaveFormat` enum at load time via Pydantic validators. Clear separation
of which formats are saved locally vs published.

---

## 8. CommonPaths Extension

**Decision**: Add `PUBLISHING_INFO_DIR` and `OUTPUT_MODELS_DIR` to `CommonPaths`. Add properties
`publishing_info` and `outputs`. Add `publishing_info` to `ensure_all_dirs_exist`.

**Rationale**: Consistent with existing pattern. `OUTPUT_MODELS_DIR` should NOT be created by
`ensure_all_dirs_exist` (it is created on demand during save); only `publishing_info` is a
config directory that should always exist.

---

## 9. Data Flow: Metrics into Model Card

**Decision**: `TrainCommand.execute()` passes `pre_run_results` and `post_run_results` (both
`list[MetricResult]`) to `publishing_helper.save_model(...)`. The helper converts these to a
dict mapping metric name → value for model card rendering.

**Rationale**: `MetricResult` is already a `NamedTuple(metric_name, metric_result)`. No new
protocol or data class needed; existing type flows cleanly through.

**Edge**: If `run_comparison_before_training=False`, `pre_run_results` will be empty; model
card omits pre-training section gracefully.

---

## 10. PublishCommand Structure

**Decision**: New `PublishCommand` dataclass in `commands/publish.py` implementing
`CommmandProtocol`. Takes `training_info: TrainingInfo` and `publishing_info: PublishingInfo`.
Delegates to `publishing_helper.publish_model(...)`.

CLI signature:
```
classification-trainer publish
  --training-info <name>
  --publishing-info <name>
```

**Rationale**: Consistent with existing command pattern. Train info is needed to derive the
model name and HF username (both live in `TrainingInfo`).

---

## Summary of New Dependencies

| Dependency | Required for | Install |
|------------|-------------|---------|
| `autoawq` | AWQ format save | Optional (`[awq]` extra) |
| `huggingface_hub` | HF upload + ModelCard | Already transitive via `transformers` |

No mandatory new dependencies for GGUF, LoRA, or merged formats — all covered by Unsloth
and existing HuggingFace stack.
