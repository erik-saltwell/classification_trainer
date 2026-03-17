# Contract: Generated Modelfile Format

This document defines the exact structure of the `Modelfile` artifact generated for
GGUF and merged models during the save/publish workflow.

## File Identification

- **Filename**: `Modelfile` (no extension, exact case)
- **Encoding**: UTF-8
- **Location**: root of the format-specific save directory, alongside `README.md`

## Structure

```
FROM <model_ref>

SYSTEM """
<verbatim system prompt>
"""

TEMPLATE """
<go_template>
"""

PARAMETER temperature <float>
PARAMETER top_p <float>
PARAMETER num_predict <int>
PARAMETER num_ctx <int>
[PARAMETER repeat_penalty <float>]
PARAMETER stop "<stop_string>"
[... one PARAMETER stop line per additional stop string ...]
```

## Section Specifications

### FROM

- **GGUF format**: relative path to the primary quantization file.
  Pattern: `<model_name>-gguf-<quant>.gguf`
  Example: `FROM mymodel-gguf-q8_0.gguf`
  The primary quantization is the first entry in `PublishingInfo.gguf_quantizations`.

- **Merged format**: HuggingFace repo ID.
  Pattern: `<hugging_face_user_name>/<model_name>-merged`
  Example: `FROM eriksalt/my-classifier-merged`

### SYSTEM

The verbatim system prompt from `TrainingInfo.system_prompt`, wrapped in triple quotes.
Newlines and special characters are preserved as-is; no escaping applied.

```
SYSTEM """
<content of system prompt, may be multi-line>
"""
```

### TEMPLATE

A Go `text/template` string that reconstructs the chat format used during training.
Derived from `ChatTemplateInfo` fields. Two variants:

**With system_separator** (e.g., ChatML, Llama):
```
{{ if .System }}<system_separator>{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<instruction_separator>{{ .Prompt }}<end_of_turn>
{{ end }}<response_separator>
```

**Without system_separator** (e.g., Mistral — inline system before instruction):
```
{{ if .System }}{{ .System }}
{{ end }}{{ if .Prompt }}<instruction_separator>{{ .Prompt }}<end_of_turn>
{{ end }}<response_separator>
```

Where `<end_of_turn>` = `stop_strings[0]` from `ChatTemplateInfo`.

### PARAMETER blocks

All parameters appear on separate lines in a fixed order:

1. `PARAMETER temperature` — always present (`InferenceInfo.temperature`)
2. `PARAMETER top_p` — always present (`InferenceInfo.top_p`)
3. `PARAMETER num_predict` — always present (`InferenceInfo.max_new_tokens`)
4. `PARAMETER num_ctx` — always present (`TrainingInfo.max_sequence_length`)
5. `PARAMETER repeat_penalty` — **only present if** `InferenceInfo.repetition_penalty` is not null
6. `PARAMETER stop "<value>"` — one line per entry in `ChatTemplateInfo.stop_strings`; omitted entirely if empty

## Example (ChatML, GGUF, classification config)

```
FROM my-rpg-classifier-gguf-q8_0.gguf

SYSTEM """
You are a binary classifier. Read the following Reddit post and classify it as
"positive" (related to tabletop RPG) or "negative" (not related).
Respond with only the label.
"""

TEMPLATE """
{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""

PARAMETER temperature 0.0
PARAMETER top_p 1.0
PARAMETER num_predict 8
PARAMETER num_ctx 1024
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
```

## Invariants

- The `FROM` line is always the first non-blank line.
- `SYSTEM`, `TEMPLATE`, and all `PARAMETER` blocks are always present (except `repeat_penalty` and empty stop lists).
- Parameter order is fixed as listed above.
- No blank lines within `PARAMETER` block.
- One blank line separates each top-level section.
