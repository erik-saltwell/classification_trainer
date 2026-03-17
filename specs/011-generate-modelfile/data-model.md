# Data Model: Modelfile Generation on Publish

## Modified Entities

### ChatTemplateInfo (configuration/chat_template_info.py)

**Existing fields** (unchanged):
- `instruction_separator: str` — token sequence opening the user turn (e.g., `<|im_start|>user\n`)
- `response_separator: str` — token sequence opening the assistant turn (e.g., `<|im_start|>assistant\n`)
- `stop_strings: tuple[str, ...] = ()` — generation stop sequences
- `eos_token_strings: tuple[str, ...]` — EOS token strings for the tokenizer
- `add_special_tokens: bool`
- `assistant_newline: bool`

**New field**:
- `system_separator: str | None = None` — the token sequence that opens the system message turn in the prompt template. If `None`, the system prompt is rendered inline before the first user turn (Mistral style). If set, a dedicated system block is emitted.

**Validation**: No new validation needed; field is optional with a safe default.

**YAML updates required** (chat_template_info/*.yaml):

| File | system_separator value |
|---|---|
| `chat-ml.yaml` | `"<\|im_start\|>system\n"` |
| `llama.yaml` | `"<\|start_header_id\|>system<\|end_header_id\|>\n\n"` |
| `mistral.yaml` | `null` (inline system — Mistral convention) |
| `gemma.yaml` | `"<start_of_turn>system\n"` |
| `phi.yaml` | `"<\|system\|>\n"` |

---

## New Artifacts (Generated Files)

### Modelfile

A plain-text file written to each qualifying format's save directory alongside `README.md`.

**Filename**: `Modelfile` (no extension, exact case)

**Location**:
- GGUF: `outputs/saves/<model_name>/gguf/Modelfile`
- Merged: `outputs/saves/<model_name>/merged/Modelfile`
- LoRA: not generated

**Structure** (all sections always present except `repeat_penalty`):

```
FROM {model_ref}

SYSTEM """
{system_prompt}
"""

TEMPLATE """
{go_template}
"""

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER num_predict {max_new_tokens}
PARAMETER num_ctx {max_sequence_length}
[PARAMETER repeat_penalty {repetition_penalty}]   # present only if not None
PARAMETER stop "{stop_string_1}"
[PARAMETER stop "{stop_string_N}"]                # one per stop_string
```

**Field derivation table**:

| Modelfile field | Source attribute | Notes |
|---|---|---|
| `FROM` (GGUF) | `publishing_info.gguf_quantizations[0]` + naming convention | e.g., `mymodel-gguf-q8_0.gguf` |
| `FROM` (merged) | `training_info.hugging_face_user_name` + `training_info.model_name` | e.g., `username/model-merged` |
| `SYSTEM` | `training_info.system_prompt` | verbatim; triple-quoted |
| `TEMPLATE` | `chat_template_info.system_separator`, `instruction_separator`, `response_separator`, `stop_strings[0]` | Go template syntax |
| `temperature` | `inference_info.temperature` | float |
| `top_p` | `inference_info.top_p` | float |
| `num_predict` | `inference_info.max_new_tokens` | int |
| `num_ctx` | `training_info.max_sequence_length` | int |
| `repeat_penalty` | `inference_info.repetition_penalty` | omitted if None |
| `stop` (each) | each entry in `chat_template_info.stop_strings` | one PARAMETER line per entry |

---

## No New Pydantic Models

The Modelfile is a generated text artifact, not a config model. No new `BaseModel` subclass is needed. The existing config models already hold all required data; only `ChatTemplateInfo` gains one field.
