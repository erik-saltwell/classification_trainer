# Research: GGUF HuggingFace Metadata Files for Ollama

**Sources**: HuggingFace Hub Ollama docs, Ollama Modelfile reference, Ollama template docs, GitHub issues.

---

## Decision 1: What files are needed and where do they live?

**Decision**: Three plain-text files at the **root** of the GGUF HuggingFace repository (no subdirectory):

| Filename | Format | Purpose |
|---|---|---|
| `template` | Go template text (no wrapper) | Chat template Ollama uses for prompt formatting |
| `system` | Plain UTF-8 text | Default system prompt |
| `params` | JSON flat object | Default sampling parameters |

**Rationale**: HuggingFace Hub's Ollama integration reads these exact filenames at the repo root when `ollama run hf.co/user/repo` is executed. Ollama downloads the GGUF file(s) then looks for these optional metadata files to configure the model's behavior. This is documented at https://huggingface.co/docs/hub/en/ollama.

**Alternatives considered**: Subdirectory (e.g., `ollama/template`). Not used by the spec — Ollama expects root-level files only.

---

## Decision 2: These files are GGUF-only

**Decision**: `template`, `system`, and `params` are generated **only for SaveFormat.GGUF**. Merged repos use the `Modelfile` approach (FROM HF repo ID), which Ollama handles differently.

**Rationale**: The HF Hub Ollama documentation specifically describes this pattern for GGUF repositories. Merged repos (Safetensors) are imported via a different Ollama code path that reads the `Modelfile` directly.

---

## Decision 3: `template` file content

**Decision**: The `template` file contains the **raw Go template body** — the same string already computed as `template_body` in `generate_modelfile()` — without the `TEMPLATE """..."""` Modelfile wrapper.

**Example** (ChatML):
```
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{- end }}
{{- if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{- end }}
<|im_start|>assistant
```

**Rationale**: The `template` file content is identical in structure to the Modelfile TEMPLATE block body. Extracting `_build_template_body()` as a shared private function eliminates duplication between `generate_modelfile()` and the new metadata generator.

---

## Decision 4: `system` file content

**Decision**: The `system` file contains `training_info.system_prompt` written verbatim as UTF-8 text, with no wrappers, quotes, or extra newlines beyond what the prompt itself contains.

**Rationale**: Ollama reads this as raw text and uses it as the default system message. No encoding needed.

---

## Decision 5: `params` file content and format

**Decision**: The `params` file is a **JSON object** with the following keys derived from config:

```json
{
  "temperature": <InferenceInfo.temperature>,
  "top_p": <InferenceInfo.top_p>,
  "num_predict": <InferenceInfo.max_new_tokens>,
  "num_ctx": <TrainingInfo.max_sequence_length>,
  "stop": [<each entry in ChatTemplateInfo.stop_strings>]
}
```

Plus optionally:
```json
  "repeat_penalty": <InferenceInfo.repetition_penalty>
```
(omitted when `repetition_penalty` is `None`.)

**Rationale**: These are the same parameters already written as `PARAMETER` lines in the Modelfile. Using `json.dumps(..., indent=2)` from the standard library produces valid, human-readable output. Ollama's params file uses the same parameter names as the Modelfile PARAMETER keys.

**Alternatives considered**: YAML format. Rejected — Ollama's documented format is JSON. Plain `key=value`. Rejected — not what the spec describes.

---

## Decision 6: Refactor — extract `_build_template_body()`

**Decision**: Extract a private helper `_build_template_body(chat_template_info) -> str` from the existing `generate_modelfile()` function. Both `generate_modelfile()` and the new `generate_gguf_hf_metadata()` call it.

**Rationale**: Avoids duplicating the template-building logic (system_separator branching, end_of_turn derivation, instr/resp separator composition). Three lines of shared logic → one shared function. Passes constitution Principle V (simplicity — no premature abstraction, but deduplication of real duplication).

---

## Decision 7: `publish_model()` regeneration behavior

**Decision**: During `publish_model()`, for GGUF format, check if **all three** metadata files are present. If any is missing, regenerate all three via `generate_gguf_hf_metadata()` before uploading. This mirrors the existing Modelfile regeneration logic.

**Rationale**: Consistent behavior — users who publish pre-existing GGUF saves (saved before this feature) automatically get the metadata files on their next publish without re-training.

---

## Decision 8: No new dependencies

**Decision**: Use Python stdlib `json` module for params serialization. No new packages.

**Rationale**: `json` is already available; the params structure is a flat dict with no special serialization needs.
