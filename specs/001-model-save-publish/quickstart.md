# Quickstart: Model Save and Publish

**Feature**: 001-model-save-publish
**Date**: 2026-03-05

---

## Prerequisites

- A completed training run (or use the flow below which trains and saves in one step)
- HuggingFace account with a valid token (`HF_TOKEN` environment variable or `huggingface-cli login`)
- For AWQ format: GPU with sufficient VRAM (~2x base model size) and `autoawq` installed

---

## Step 1: Create a Publishing Config

Create `publishing_info/my-run.yaml`:

```yaml
description: "Fine-tuned Llama 3.2-1B-Instruct for binary classification on Reddit RPG posts."

save_formats:
  - lora
  - gguf

publish_formats:
  - lora
  - gguf

gguf_quantization: q8_0
```

---

## Step 2: Train and Save

Add `--publishing-info` to your existing train command:

```bash
python -m classification_trainer train \
  --dataset my-dataset \
  --base-model llama-3.2-1b \
  --training-info my-training-run \
  --inference-info my-inference \
  --publishing-info my-run
```

After training completes, artifacts are saved to:
```
output_models/my-classifier/
├── lora/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── README.md
└── gguf/
    ├── my-classifier-q8_0.gguf
    └── README.md
```

---

## Step 3: Inspect the Model Card

Review the generated card before publishing:

```bash
cat output_models/my-classifier/gguf/README.md
```

The card contains your description, training config, dataset info, and pre/post metrics.

---

## Step 4: Publish to HuggingFace

```bash
python -m classification_trainer publish \
  --training-info my-training-run \
  --publishing-info my-run
```

Expected output:
```
Publishing model artifacts to HuggingFace Hub...
Publishing lora → alice/my-classifier-lora  ✓
Publishing gguf → alice/my-classifier-gguf  ✓
All formats published successfully.
```

Each format is its own repository on HuggingFace:
- `https://huggingface.co/alice/my-classifier-lora`
- `https://huggingface.co/alice/my-classifier-gguf`

---

## Using Published Models

### GGUF with Ollama

```bash
ollama run alice/my-classifier-gguf
```

### LoRA with Python (PEFT)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "alice/my-classifier-lora")
tokenizer = AutoTokenizer.from_pretrained("alice/my-classifier-lora")
```

### Merged Checkpoint with HF Pipeline

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="alice/my-classifier-merged")
```

### AWQ with vLLM

```python
from vllm import LLM

llm = LLM(model="alice/my-classifier-awq")
```

---

## Save-Only (No Publish)

To save locally without publishing, set `publish_formats` to an empty list:

```yaml
# publishing_info/save-only.yaml
description: "..."
save_formats:
  - lora
  - gguf
publish_formats: []
```

Then run `train --publishing-info save-only` — no HuggingFace credentials needed.

---

## Validation

Confirm the feature works correctly:

1. `output_models/<model-name>/` directory exists with non-empty format subdirectories
2. Each format directory contains `README.md`
3. `README.md` contains the `description` text from the publishing config
4. `README.md` contains pre-training and post-training metrics
5. After `publish`, HuggingFace repositories exist at the expected URLs
6. HuggingFace repository README matches the local `README.md`
7. Running `train` without `--publishing-info` produces no `output_models/` directory
