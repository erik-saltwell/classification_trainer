# Contract: HuggingFace GGUF Metadata Files

These three files are written to the GGUF save directory and uploaded as part of the
GGUF HuggingFace repository. They enable `ollama run hf.co/<user>/<repo>` to work
without any user configuration.

## File Inventory

All files live at the **repository root** (same level as `.gguf` files):

```
<gguf-save-dir>/
├── <model>-gguf-<quant>.gguf   # model weights (existing)
├── Modelfile                    # existing Ollama Modelfile
├── README.md                    # existing model card
├── template                     # NEW: Go template for chat format
├── system                       # NEW: default system prompt
└── params                       # NEW: default sampling parameters
```

---

## `template`

**Encoding**: UTF-8, no BOM
**Format**: Go `text/template` syntax
**Content**: Raw template body (no `TEMPLATE """..."""` Modelfile wrapper)

### Structure (with system_separator set — e.g., ChatML)

```
{{- if .System }}<system_separator>{{ .System }}<end_of_turn>
{{- end }}
{{- if .Prompt }}<instruction_separator>{{ .Prompt }}<end_of_turn>
{{- end }}
<response_separator>
```

### Structure (system_separator = null — e.g., Mistral inline)

```
{{- if .System }}{{ .System }}
{{- end }}
{{- if .Prompt }}<instruction_separator>{{ .Prompt }}<end_of_turn>
{{- end }}
<response_separator>
```

### Example (ChatML)

```
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
{{- if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{- end }}
<|im_start|>assistant
```

---

## `system`

**Encoding**: UTF-8, no BOM
**Format**: Plain text
**Content**: Verbatim system prompt from `TrainingInfo.system_prompt`
**No** wrappers, quotes, or extra trailing newlines beyond the prompt itself.

---

## `params`

**Encoding**: UTF-8, no BOM
**Format**: JSON object, 2-space indent

### Schema

```json
{
  "temperature": <float>,
  "top_p": <float>,
  "num_predict": <int>,
  "num_ctx": <int>,
  "stop": ["<stop_string_1>", "..."],
  "repeat_penalty": <float>    // OPTIONAL — omitted when repetition_penalty is null
}
```

### Field mapping

| JSON key | Source | Notes |
|---|---|---|
| `temperature` | `InferenceInfo.temperature` | Always present |
| `top_p` | `InferenceInfo.top_p` | Always present |
| `num_predict` | `InferenceInfo.max_new_tokens` | Always present |
| `num_ctx` | `TrainingInfo.max_sequence_length` | Always present |
| `stop` | `ChatTemplateInfo.stop_strings` | Always present; empty array `[]` when no stops |
| `repeat_penalty` | `InferenceInfo.repetition_penalty` | **Omitted** when null |

### Example

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "num_predict": 8,
  "num_ctx": 1024,
  "stop": [
    "<|im_end|>",
    "<|im_start|>"
  ]
}
```

---

## Invariants

- All three files are always generated together for GGUF format; never generated for LORA or MERGED.
- Files are overwritten unconditionally on re-save (same behavior as `Modelfile` and `README.md`).
- `params` key order is fixed: temperature, top_p, num_predict, num_ctx, stop, repeat_penalty (if present).
- `stop` is always a JSON array even when empty.
