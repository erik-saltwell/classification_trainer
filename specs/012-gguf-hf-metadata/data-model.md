# Data Model: GGUF HuggingFace Metadata Files

## No New Configuration Models

All content for the three metadata files is derived from existing Pydantic models:
- `TrainingInfo` — `system_prompt`, `max_sequence_length`, `model_name`, `hugging_face_user_name`
- `InferenceInfo` — `temperature`, `top_p`, `max_new_tokens`, `repetition_penalty`
- `ChatTemplateInfo` — `stop_strings`, `instruction_separator`, `response_separator`, `system_separator`
- `PublishingInfo` — `gguf_quantizations`

No new fields, models, or YAML files are needed.

---

## New Code Entities

### `_build_template_body(chat_template_info: ChatTemplateInfo) -> str`

**Location**: `helpers/publishing_helper.py` (private function)

**Purpose**: Produces the Go template string used in both the Modelfile TEMPLATE block and the standalone `template` file. Extracted from `generate_modelfile()` to eliminate duplication.

**Logic** (identical to existing template_body construction):
- `end_of_turn = stop_strings[0]` if stop_strings else `""`
- If `system_separator` is not None: emit system block with separator
- If `system_separator` is None: emit inline system block
- Always emit prompt block and response_separator

---

### `generate_gguf_hf_metadata(save_dir, training_info, publishing_info) -> None`

**Location**: `helpers/publishing_helper.py`

**Purpose**: Writes the three HuggingFace Ollama metadata files into `save_dir`. Called only for `SaveFormat.GGUF`.

**Outputs** (all at repo root / `save_dir`):

| File | Content | Source |
|---|---|---|
| `template` | Raw Go template body from `_build_template_body()` | `ChatTemplateInfo` |
| `system` | Verbatim system prompt | `TrainingInfo.system_prompt` |
| `params` | JSON object with inference parameters | `InferenceInfo` + `TrainingInfo` |

---

## New Artifacts

### `template` file

- **Filename**: `template` (no extension)
- **Format**: Go template text, UTF-8
- **Location**: root of GGUF save directory (same level as `.gguf` files and `Modelfile`)

### `system` file

- **Filename**: `system` (no extension)
- **Format**: Plain UTF-8 text
- **Location**: root of GGUF save directory

### `params` file

- **Filename**: `params` (no extension)
- **Format**: JSON object, UTF-8, 2-space indent
- **Location**: root of GGUF save directory
- **Schema**:

```json
{
  "temperature": float,
  "top_p": float,
  "num_predict": int,
  "num_ctx": int,
  "stop": [string, ...],
  "repeat_penalty": float   // optional — only present if repetition_penalty != null
}
```
