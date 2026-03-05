# CLI Contracts: Model Save and Publish

**Feature**: 001-model-save-publish
**Date**: 2026-03-05

---

## Modified Command: `train`

Adds an optional `--publishing-info` argument. All existing arguments and behavior are
unchanged when `--publishing-info` is omitted.

```
classification-trainer train
  --dataset         <dataset-info-name>
  --base-model      <base-model-info-name>
  --training-info   <training-info-name>
  --inference-info  <inference-info-name>
  [--publishing-info <publishing-info-name>]      # NEW: optional
  [--run-comparison-before-training]
```

**`--publishing-info`** (optional, default: none)
- YAML filename without extension from `publishing_info/` directory
- When omitted: no saving occurs; training behaves exactly as before
- When provided: model is saved to disk in all `save_formats` after training completes

**Exit codes**:
- `0`: Success
- `1`: Config validation error (including invalid publishing config)
- `1`: Save failure (disk full, etc.)

**Output (when --publishing-info provided)**:
```
[blue]Saving model artifacts...[/blue]
Saving lora → output_models/<model-name>/lora/  ✓
Saving gguf → output_models/<model-name>/gguf/
    Quantizing q8_0 → <model-name>-gguf-q8_0.gguf
    Quantizing q4_k_m → <model-name>-gguf-q4_k_m.gguf
  ✓ gguf
```

---

## New Command: `publish`

Uploads locally saved model artifacts to HuggingFace Hub. One repository per format.

```
classification-trainer publish
  --training-info   <training-info-name>
  --publishing-info <publishing-info-name>
```

**`--training-info`** (required)
- YAML filename without extension from `training_info/` directory
- Used to derive the model name and HuggingFace username

**`--publishing-info`** (required)
- YAML filename without extension from `publishing_info/` directory
- Specifies which formats to publish and the description

**Behavior**:
1. Load and validate both configs (fail fast on invalid config)
2. For each enabled publish format (per boolean flags), including one iteration per GGUF quant:
   a. Check that `output_models/<model-name>/<format-slug>/` exists and contains `README.md`
   b. Create HuggingFace repository `<hf-username>/<model-name>-<format-slug>` if not exists (private)
   c. Upload all files in the format directory including `README.md`
   d. Report success or failure for this format
3. Exit 0 only if all formats succeeded; exit 1 if any format failed

**Exit codes**:
- `0`: All configured publish formats uploaded successfully
- `1`: Config validation error
- `1`: One or more format uploads failed (partial success still exits 1)

**Output**:
```
[blue]Publishing model artifacts to HuggingFace Hub...[/blue]
Publishing lora → alice/my-classifier-lora  ✓
Publishing gguf → alice/my-classifier-gguf  ✓
[green]All formats published successfully.[/green]
```

**Error examples**:
```
[red]Error: No saved artifacts found at output_models/my-classifier/gguf-q8_0/.
Run `train --publishing-info <name>` first.[/red]

[red]Error: Missing model card at output_models/my-classifier/lora/README.md.
Re-run `train --publishing-info <name>` to regenerate.[/red]
```

---

## PublishingInfo YAML Schema

```yaml
# publishing_info/<name>.yaml

# Required: Primary description for all model cards
description: "string"

# Optional: GGUF quantization levels to produce (default: [q8_0])
# Each level becomes its own directory and HuggingFace repository.
gguf_quantizations:
  - q8_0
  - q4_k_m

# Optional: Unsloth merged save method (default: merged_16bit)
merged_save_method: merged_16bit

# --- Save to local disk (all default false) ---
save_gguf: true
save_lora: true
save_merged: false

# --- Publish to HuggingFace Hub (all default false) ---
publish_gguf: true
publish_lora: false
publish_merged: false
```

**Validation errors** (raised at startup before any model work):
- Missing required field (`description`)

---

## Output Directory Layout Contract

```
output_models/
└── <TrainingInfo.model_name>/
    ├── gguf/                              # all GGUF quants share one directory
    │   ├── <model>-gguf-<quant>.gguf     # one file per quantization level
    │   └── README.md                      # MUST exist; absence blocks publish
    └── <format-slug>/                     # lora / merged each get own dir
        ├── <model files>
        └── README.md                      # MUST exist; absence blocks publish
```

The `publish` command treats the presence of `README.md` as a sentinel confirming the
save completed successfully.

---

## HuggingFace Repository Naming Contract

| Format | Format slug | Repository ID | Files in repo |
|--------|------------|--------------|---------------|
| GGUF (all quants) | `gguf` | `alice/my-classifier-gguf` | `my-classifier-gguf-q8_0.gguf`, `my-classifier-gguf-q4_k_m.gguf`, `README.md` |
| LoRA adapter | `lora` | `alice/my-classifier-lora` | adapter files, `README.md` |
| Merged checkpoint | `merged` | `alice/my-classifier-merged` | safetensors, `README.md` |

Pattern: `<TrainingInfo.hugging_face_user_name>/<TrainingInfo.model_name>-<format-slug>`

GGUF file naming pattern: `<TrainingInfo.model_name>-gguf-<quant>.gguf`
(e.g. `my-classifier-gguf-q8_0.gguf`, `my-classifier-gguf-q4_k_m.gguf`)
