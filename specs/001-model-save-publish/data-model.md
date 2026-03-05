# Data Model: Model Save and Publish

**Feature**: 001-model-save-publish
**Date**: 2026-03-05

---

## Entities

### SaveFormat (Internal Enum)

Internal slug enum used by helper code for all formats. Not user-facing.

| Slug | Value | Target | Notes |
|------|-------|--------|-------|
| `gguf` | `"gguf"` | Ollama / llama.cpp | All quants share one dir; one HF repo |
| `lora` | `"lora"` | PEFT/HF | Adapter only (no base weights) |
| `merged` | `"merged"` | HF pipeline / vLLM | Full merged 16-bit checkpoint |

All GGUF quantization levels share the single slug `gguf`. Within the `gguf/`
directory, each quantization is a separate file named
`<model-name>-gguf-<quant>.gguf`. The list of quantizations is driven by
`PublishingInfo.gguf_quantizations`.

---

### PublishingInfo (Pydantic Model)

Loaded from `publishing_info/<name>.yaml`. Controls which formats are saved, which are
published, and model card content.

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `description` | `str` | Yes | — | Appears verbatim as the primary description in every model card |
| `gguf_quantizations` | `list[str]` | No | `["q8_0"]` | GGUF quant levels to produce; each becomes its own artifact |
| `merged_save_method` | `str` | No | `"merged_16bit"` | Unsloth save method; e.g., `"merged_4bit_forced"` |
| `save_gguf` | `bool` | No | `false` | Save GGUF artifact(s) to disk |
| `save_lora` | `bool` | No | `false` | Save LoRA adapter to disk |
| `save_merged` | `bool` | No | `false` | Save merged HF checkpoint to disk |
| `publish_gguf` | `bool` | No | `false` | Upload GGUF artifact(s) to HuggingFace |
| `publish_lora` | `bool` | No | `false` | Upload LoRA adapter to HuggingFace |
| `publish_merged` | `bool` | No | `false` | Upload merged checkpoint to HuggingFace |

**Rules**:
- All boolean flags default to `false`; a config with all flags false produces no artifacts.
- `gguf_quantizations` applies whenever `save_gguf` or `publish_gguf` is `true`.
- `gguf_quantization` strings are validated at runtime by Unsloth (not by Pydantic).
- A format may be published without being saved locally; the system saves transiently.

**YAML location**: `publishing_info/<name>.yaml`

**Example**:
```yaml
description: "Fine-tuned Llama 3.2-1B for binary classification on Reddit RPG posts."
gguf_quantizations:
  - q8_0
  - q4_k_m
save_gguf: true
save_lora: true
publish_gguf: true
publish_lora: true
```

---

### SavedModelArtifact (Runtime concept — not a Pydantic model)

Represents the collection of files produced by saving one format to disk.

**Location**: `output_models/<hf-model-name>/<format-slug>/`
- `hf-model-name` = `TrainingInfo.model_name` (the bare model name, not `username/model`)
- `format-slug` = `gguf`, `lora`, or `merged`

**Contents per format**:

| Format slug | Files |
|------------|-------|
| `gguf` | `<model-name>-gguf-<quant>.gguf` (one per quant level), `README.md` |
| `lora` | `adapter_config.json`, `adapter_model.safetensors` (or `.bin`), `tokenizer.*`, `README.md` |
| `merged` | `config.json`, `model-*.safetensors`, `tokenizer.*`, `special_tokens_map.json`, `README.md` |

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

- **gguf**: List of available files (`<model-name>-gguf-<quant>.gguf`), llama.cpp load example
- **lora**: PEFT `PeftModel.from_pretrained(base_model, repo_id)` Python snippet
- **merged**: `AutoModelForCausalLM.from_pretrained(repo_id)` Python snippet

---

### HuggingFaceRepository (External — per format)

One repository per saved format on HuggingFace Hub.

**Naming**: `<hf-username>/<model-name>-<format-slug>`
- `hf-username` = `TrainingInfo.hugging_face_user_name`
- `model-name` = `TrainingInfo.model_name`
- `format-slug` = `gguf`, `lora`, or `merged`

**Examples** (with `gguf_quantizations: [q8_0, q4_k_m]`):
- `alice/my-classifier-gguf` — contains `my-classifier-gguf-q8_0.gguf` and `my-classifier-gguf-q4_k_m.gguf`
- `alice/my-classifier-lora`
- `alice/my-classifier-merged`

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
    │   ├── <model>-gguf-q8_0.gguf
    │   ├── <model>-gguf-q4_k_m.gguf
    │   └── README.md
    ├── lora/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   ├── tokenizer.json
    │   └── README.md
    └── merged/
        ├── config.json
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
