# Research: Modelfile Generation on Publish

## Decision 1: Which formats receive a Modelfile

**Decision**: GGUF and merged formats only. LoRA adapters receive no Modelfile.

**Rationale**: Ollama's primary format is GGUF. Merged Safetensors are importable via `FROM hf.co/...`. LoRA adapters in Ollama require a matching base model to be loaded first and the ADAPTER instruction support is limited and version-dependent — generating a broken Modelfile is worse than generating none.

**Alternatives considered**: Generating a Modelfile for LoRA using the ADAPTER instruction. Rejected because Ollama only supports one GGUF adapter per model and Safetensors LoRA support varies by version; a LoRA Modelfile would work in narrow conditions and mislead users elsewhere.

---

## Decision 2: FROM instruction value

**Decision**:
- GGUF: relative filename of the primary (first) quantization — e.g., `FROM mymodel-gguf-q8_0.gguf`
- Merged: HuggingFace repo ID — e.g., `FROM username/model-name-merged`

**Rationale**: GGUF Modelfiles are used locally (user downloads both files); a relative path works when both are in the same directory. Merged models are imported via Ollama's HuggingFace integration (`ollama run hf.co/...`); the repo ID enables online import without modification. Local users of merged models can trivially change one line.

**Alternatives considered**: `FROM .` for merged (points to current directory). Rejected because it only works if the entire folder is downloaded and Ollama is invoked from that directory — fragile. Using the repo ID aligns with the primary use case.

---

## Decision 3: SYSTEM instruction encoding

**Decision**: Use Ollama triple-quote format for all system prompts.

```
SYSTEM """
{system_prompt_text}
"""
```

**Rationale**: System prompts in this project are multi-line (they include structured classification instructions). Triple-quote handles embedded newlines, quotes, and special characters without escaping. Single-quote strings would break on any prompt containing `"` characters.

**Alternatives considered**: Single-quoted `SYSTEM "..."`. Rejected because prompts contain newlines and may contain double quotes.

---

## Decision 4: TEMPLATE instruction — deriving a Go template from ChatTemplateInfo

**Decision**: Add an optional `system_separator: str | None` field to `ChatTemplateInfo`. This is the only addition to existing configuration models required by this feature.

The Go template is constructed as follows:

```
{{ if .System }}{system_separator}{{ .System }}{end_of_turn}\n{{ end }}{{ if .Prompt }}{instruction_separator}{{ .Prompt }}{end_of_turn}\n{{ end }}{response_separator}
```

Where:
- `system_separator` — from the new `ChatTemplateInfo.system_separator` field (if None, system prompt is rendered inline before the instruction)
- `end_of_turn` — `stop_strings[0]` (the primary EOS/end-of-turn marker)
- `instruction_separator` — existing `ChatTemplateInfo.instruction_separator`
- `response_separator` — existing `ChatTemplateInfo.response_separator`

**Existing template configs updated with `system_separator`**:

| Template | system_separator |
|---|---|
| chat-ml | `<\|im_start\|>system\n` |
| llama | `<\|start_header_id\|>system<\|end_header_id\|>\n\n` |
| mistral | `None` (system inline before `[INST]`) |
| gemma | `<start_of_turn>system\n` |
| phi | `<\|system\|>\n` |

**Rationale**: The `ChatTemplateInfo` already holds all the pieces of the prompt format. The only missing piece is how to wrap the system message, which varies across architectures. Adding one optional nullable field is the minimal config-first change that makes the template derivation fully data-driven with no hardcoded template name checks.

**Alternatives considered**:
1. Add a `get_ollama_template() -> str` method to ChatTemplateInfo. Rejected — methods with return values belong in helpers per Principle III; ChatTemplateInfo is config-only.
2. Hardcode template logic per known template name. Rejected — violates Principle I (config-first) and breaks for user-added templates.
3. Add a full `ollama_template: str` field to ChatTemplateInfo. Rejected as over-engineering; the template is derivable from existing fields + one small addition.

---

## Decision 5: PARAMETER mappings

| Modelfile PARAMETER | Source | Notes |
|---|---|---|
| `temperature` | `InferenceInfo.temperature` | Always included |
| `top_p` | `InferenceInfo.top_p` | Always included |
| `num_predict` | `InferenceInfo.max_new_tokens` | Always included |
| `num_ctx` | `TrainingInfo.max_sequence_length` | Always included; ensures context matches training |
| `repeat_penalty` | `InferenceInfo.repetition_penalty` | Omitted if None |
| `stop` | `ChatTemplateInfo.stop_strings` | One `PARAMETER stop "..."` line per entry |

**Rationale**: These are the parameters that affect output quality and must match the conditions under which the model was evaluated. The remaining Ollama parameters (seed, top_k, mirostat, etc.) have no counterpart in training config and are left at Ollama defaults.

---

## Decision 6: Code placement

**Decision**: New function `generate_modelfile()` lives in `helpers/publishing_helper.py`, the existing module for all save/publish domain logic.

Called from two sites:
1. `_save_format()` in `save_model()` — invoked immediately after `generate_model_card()`, inside the same try/except that cleans up on failure. This means Modelfile failure fails the whole format save.
2. `publish_model()` — checks for `Modelfile` presence before uploading; generates if missing.

**Rationale**: Keeps all save/publish domain logic in one helper. No new module needed. Consistent with Principle III: helper handles domain logic, command (`PublishCommand`) does orchestration only.

---

## Decision 7: No new dependencies

**Decision**: Modelfile generation uses only Python standard library string formatting. No new packages.

**Rationale**: The Modelfile format is plain text with no serialization complexity. Standard f-strings and `Path.write_text()` are sufficient.
