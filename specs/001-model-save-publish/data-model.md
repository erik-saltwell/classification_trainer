# Data Model: Model Save and Publish

**Feature**: 001-model-save-publish
**Date**: 2026-03-05

---

## Entities

### SaveFormat (Enum)

Represents one of the four supported output formats.

| Slug | Value | Target | Notes |
|------|-------|--------|-------|
| `gguf` | `"gguf"` | Ollama | Quantization level configurable |
| `lora` | `"lora"` | PEFT/HF | Adapter only (no base weights) |
| `merged` | `"merged"` | HF pipeline / vLLM | Full merged 16-bit checkpoint |
| `awq` | `"awq"` | vLLM | AWQ-quantized merged model |

**Constraints**:
- Must be one of these four values; any other string fails Pydantic validation at config load time.

---

### PublishingInfo (Pydantic Model)

Loaded from `publishing_info/<name>.yaml`. Controls which formats are saved, which are published,
HuggingFace destination, and model card description.

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `description` | `str` | Yes | — | Appears verbatim as the primary description in every model card |
| `save_formats` | `list[SaveFormat]` | Yes | — | Formats to save to disk after training; may be empty list (no save) |
| `publish_formats` | `list[SaveFormat]` | Yes | — | Formats to upload to HuggingFace Hub |
| `gguf_quantization` | `str` | No | `"q8_0"` | Quantization method for GGUF; e.g., `"q4_k_m"`, `"f16"` |
| `merged_save_method` | `str` | No | `"merged_16bit"` | Unsloth save method for merged format; e.g., `"merged_4bit_forced"` |

**Validation rules**:
- `save_formats` and `publish_formats` elements MUST be valid `SaveFormat` enum values.
- A format listed in `publish_formats` but not in `save_formats` is saved transiently during
  publish (not retained locally); this is acceptable.
- `gguf_quantization` is a free string (Unsloth validates it at runtime).

**YAML location**: `publishing_info/<name>.yaml`

**Example**:
```yaml
description: "Fine-tuned Llama 3.2-1B for binary classification on Reddit RPG posts."
save_formats:
  - lora
  - gguf
publish_formats:
  - lora
  - gguf
gguf_quantization: q8_0
```

---

### SavedModelArtifact (Runtime concept — not a Pydantic model)

Represents the collection of files produced by saving one format to disk.

**Location**: `output_models/<hf-model-name>/<format-slug>/`
- `hf-model-name` = `TrainingInfo.model_name` (the bare model name, not `username/model`)
- `format-slug` = one of `gguf`, `lora`, `merged`, `awq`

**Contents per format**:

| Format | Files |
|--------|-------|
| `gguf` | `<model-name>-<quant>.gguf`, `README.md` |
| `lora` | `adapter_config.json`, `adapter_model.safetensors` (or `.bin`), `tokenizer.*`, `README.md` |
| `merged` | `config.json`, `model-*.safetensors`, `tokenizer.*`, `special_tokens_map.json`, `README.md` |
| `awq` | `config.json`, `model-*.safetensors`, `quant_config.json`, `tokenizer.*`, `README.md` |

**Invariant**: A `README.md` model card MUST be present in every format directory. Its absence
causes the publish command to fail with an actionable error.

---

### ModelCard (Generated File)

A Markdown `README.md` file generated per saved format. Written using `huggingface_hub.ModelCard`.

**Sections** (in order):

| Section | Source |
|---------|--------|
| Title | `{model_name} ({format})` |
| Description | `PublishingInfo.description` |
| Model Details | Base model name (`BaseModelInfo.huggingface_name`), LoRA rank, quantization |
| Dataset | `DatasetInfo.huggingface_name`, split names, positive class |
| Training Configuration | Epochs/steps, batch size, LR, max sequence length (from `TrainingInfo`) |
| Pre-Training Metrics | `list[MetricResult]` captured before training; omitted if empty |
| Post-Training Metrics | `list[MetricResult]` captured after training |
| Usage | Format-specific instructions (see below) |

**Format-specific usage instructions**:

- **gguf**: `ollama run <username>/<model-name>-gguf` (if published); local run command
- **lora**: PEFT `PeftModel.from_pretrained(base_model, repo_id)` Python snippet
- **merged**: `AutoModelForCausalLM.from_pretrained(repo_id)` Python snippet
- **awq**: vLLM `LLM(model="<username>/<model-name>-awq")` Python snippet

---

### HuggingFaceRepository (External — per format)

One repository per saved format on HuggingFace Hub.

**Naming**: `<hf-username>/<model-name>-<format-slug>`
- `hf-username` = `TrainingInfo.hugging_face_user_name`
- `model-name` = `TrainingInfo.model_name`
- `format-slug` = `SaveFormat` slug value

**Examples**:
- `alice/my-classifier-gguf`
- `alice/my-classifier-lora`
- `alice/my-classifier-merged`
- `alice/my-classifier-awq`

**Properties**: Created private by default; `exist_ok=True` (idempotent creation).

---

## State Transitions

### Model Artifact Lifecycle

```
[Training Complete]
       │
       ▼
[Save to disk: output_models/<name>/<format>/]  ← publishing_helper.save_model()
       │  (includes README.md model card)
       ▼
[Locally available]
       │
       ▼  (publish command)
[Uploaded to HuggingFace Hub]  ← publishing_helper.publish_model()
       │  (uses saved README.md — not regenerated)
       ▼
[Published on HuggingFace Hub]
```

### Error States

- **Save fails (disk full)**: Partially-written files cleaned up; error surfaced to user; subsequent formats are attempted (fail-fast per format, not per run).
- **Save fails (VRAM OOM for AWQ)**: Error surfaced with actionable message; other formats unaffected.
- **Publish fails (auth error)**: All formats fail; user prompted to set `HF_TOKEN`.
- **Publish fails (missing model card)**: That format fails with error; other formats continue.

---

## CommonPaths Extensions

| Property | Path | Created by |
|----------|------|-----------|
| `publishing_info` | `publishing_info/` | `ensure_all_dirs_exist()` |
| `output_models` | `output_models/` | `publishing_helper.save_model()` (on demand) |

---

## File Locations Summary

```
publishing_info/
└── <name>.yaml                  # PublishingInfo config

output_models/
└── <model-name>/
    ├── gguf/
    │   ├── <model>.gguf
    │   └── README.md
    ├── lora/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer.json
    │   └── README.md
    ├── merged/
    │   ├── config.json
    │   ├── model-*.safetensors
    │   ├── tokenizer.json
    │   └── README.md
    └── awq/
        ├── config.json
        ├── quant_config.json
        ├── model-*.safetensors
        ├── tokenizer.json
        └── README.md

src/classification_trainer/
├── configuration/
│   └── publishing_info.py       # PublishingInfo + SaveFormat + load_publishing_info
├── helpers/
│   └── publishing_helper.py     # save_model(), generate_model_card(), publish_model()
└── commands/
    └── publish.py               # PublishCommand
```
