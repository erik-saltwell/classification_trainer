# Implementation Plan: Modelfile Generation on Publish

**Branch**: `011-generate-modelfile` | **Date**: 2026-03-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/011-generate-modelfile/spec.md`

## Summary

When a model is saved or published, generate an Ollama-compatible `Modelfile` in the save directory for GGUF and merged formats. The Modelfile encodes inference parameters (temperature, top_p, stop sequences, context length, token limit) and the chat template from the training configuration, ensuring that anyone who downloads the model and Modelfile gets the same inference behavior observed during training evaluation. The implementation adds one optional field (`system_separator`) to `ChatTemplateInfo`, one new function (`generate_modelfile`) in `publishing_helper.py`, and updates all five existing chat template YAML files.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Pydantic v2, PyYAML, standard library only (no new packages)
**Storage**: Plain text file written to existing save directories
**Testing**: pytest (unit tests only — no GPU/HuggingFace access required)
**Target Platform**: Linux (same as existing project)
**Project Type**: CLI tool / library
**Performance Goals**: N/A — file generation is negligible overhead
**Constraints**: No new configuration flags; no new dependencies; no new directories
**Scale/Scope**: One Modelfile per qualifying format per model publish

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Configuration-First | ✅ PASS | All Modelfile content derived from existing Pydantic config models. One new optional field (`system_separator`) added to `ChatTemplateInfo` YAML schema. No hardcoded values. |
| II. Protocol-Based Interfaces | ✅ PASS | `generate_modelfile()` accepts `LoggingProtocol` for console output. No new protocols needed. |
| III. Separation of Concerns | ✅ PASS | New function lives in `helpers/publishing_helper.py` (domain logic). Commands remain orchestration-only. No cross-helper calls. |
| IV. Observability | ✅ PASS | Modelfile generation logged via `LoggingProtocol`. Failure propagates as exception (same as model card). |
| V. Simplicity & Scope | ✅ PASS | Model publish/save is within scope. One new helper function, one new config field. No new abstraction layers. |

**Post-Phase 1 re-check**: No violations. `ChatTemplateInfo` addition of `system_separator` is a minimal config-first change per Principle I.

## Project Structure

### Documentation (this feature)

```text
specs/011-generate-modelfile/
├── plan.md               # This file
├── research.md           # Phase 0: decisions on template construction, format, parameters
├── data-model.md         # Phase 1: ChatTemplateInfo change + Modelfile artifact spec
├── contracts/
│   └── modelfile-format.md   # Modelfile structure contract
└── tasks.md              # Phase 2 output (/speckit.tasks command)
```

### Source Code Changes

```text
# Modified files
src/classification_trainer/
├── configuration/
│   └── chat_template_info.py          # Add system_separator: str | None = None field
├── helpers/
│   └── publishing_helper.py           # Add generate_modelfile(); call from _save_format() + publish_model()

# New test file
tests/unit/
└── test_modelfile_generation.py       # Unit tests for generate_modelfile()

# Updated YAML config files (no code change)
chat_template_info/
├── chat-ml.yaml                       # Add system_separator: "<|im_start|>system\n"
├── llama.yaml                         # Add system_separator: "<|start_header_id|>system<|end_header_id|>\n\n"
├── mistral.yaml                       # Add system_separator: null
├── gemma.yaml                         # Add system_separator: "<start_of_turn>system\n"
└── phi.yaml                           # Add system_separator: "<|system|>\n"
```

**Structure Decision**: Single-project layout (Option 1). All changes stay within the existing `helpers/` and `configuration/` layers. No new directories or modules.

## Phase 0: Research

*Complete. See [research.md](research.md) for full findings.*

Key decisions resolved:
- GGUF and merged formats receive Modelfiles; LoRA does not
- GGUF `FROM` = relative filename of first quantization
- Merged `FROM` = HuggingFace repo ID
- `SYSTEM` uses Ollama triple-quote syntax for multi-line prompts
- `TEMPLATE` derived from `ChatTemplateInfo` + new `system_separator` field
- All 5 chat template YAMLs updated with concrete `system_separator` values
- No new Python dependencies required

## Phase 1: Design

### Data Model

*See [data-model.md](data-model.md) for full spec.*

**Single change to existing config model**:

`ChatTemplateInfo` gains one optional field:
```python
system_separator: str | None = None
```

This is the only schema change. All five chat template YAML files gain this field.

**Modelfile artifact** (see [contracts/modelfile-format.md](contracts/modelfile-format.md)):
- Plain text, `Modelfile` filename, UTF-8
- Sections: `FROM`, `SYSTEM`, `TEMPLATE`, `PARAMETER` block, `PARAMETER stop` lines

### generate_modelfile() — Function Design

**Location**: `helpers/publishing_helper.py`

**Signature**:
```python
def generate_modelfile(
    save_dir: Path,
    format_slug: str,
    training_info: TrainingInfo,
    publishing_info: PublishingInfo,
) -> None:
```

**Logic**:

1. **Build `FROM` line**:
   - GGUF: `f"FROM {model_name}-gguf-{quant}.gguf"` where `quant = publishing_info.gguf_quantizations[0]` and `model_name = training_info.model_name`
   - Merged: `f"FROM {training_info.hugging_face_user_name}/{training_info.model_name}-merged"`

2. **Build `SYSTEM` block**:
   ```python
   f'SYSTEM """\n{training_info.system_prompt}\n"""'
   ```

3. **Build `TEMPLATE` block** from `ChatTemplateInfo`:
   - `chat_template_info = training_info.base_model_info.chat_template_info`
   - `end_of_turn = chat_template_info.stop_strings[0] if chat_template_info.stop_strings else ""`
   - If `system_separator` is not None:
     ```
     {{ if .System }}{system_separator}{{ .System }}{end_of_turn}\n{{ end }}{{ if .Prompt }}{instruction_separator}{{ .Prompt }}{end_of_turn}\n{{ end }}{response_separator}
     ```
   - If `system_separator` is None:
     ```
     {{ if .System }}{{ .System }}\n{{ end }}{{ if .Prompt }}{instruction_separator}{{ .Prompt }}{end_of_turn}\n{{ end }}{response_separator}
     ```
   - Wrapped in `TEMPLATE """\n...\n"""`

4. **Build `PARAMETER` lines**:
   ```python
   inference_info = training_info.inference_info
   lines = [
       f"PARAMETER temperature {inference_info.temperature}",
       f"PARAMETER top_p {inference_info.top_p}",
       f"PARAMETER num_predict {inference_info.max_new_tokens}",
       f"PARAMETER num_ctx {training_info.max_sequence_length}",
   ]
   if inference_info.repetition_penalty is not None:
       lines.append(f"PARAMETER repeat_penalty {inference_info.repetition_penalty}")
   for stop in chat_template_info.stop_strings:
       lines.append(f'PARAMETER stop "{stop}"')
   ```

5. **Assemble and write**:
   ```python
   content = "\n\n".join([from_line, system_block, template_block, "\n".join(param_lines)])
   (save_dir / "Modelfile").write_text(content, encoding="utf-8")
   ```

### Call Sites

**In `save_model()` → `_save_format()` inner function** (after `generate_model_card()`):
```python
generate_modelfile(save_dir, slug, training_info, publishing_info)
```
This is inside the existing `try/except` that calls `shutil.rmtree(save_dir)` on failure — no additional error handling needed; Modelfile generation failure automatically fails the format save.

**In `publish_model()`** (before `api.upload_folder()`):
```python
modelfile_path = save_dir / "Modelfile"
if not modelfile_path.exists() and slug in (SaveFormat.GGUF, SaveFormat.MERGED):
    generate_modelfile(save_dir, slug, training_info, publishing_info)
```
If generation fails here, the exception propagates into the existing `except Exception as exc` block that appends the slug to `failures`.

### Test Design

**File**: `tests/unit/test_modelfile_generation.py`

Tests use minimal fake config objects (no GPU, no HuggingFace, no file I/O except in-memory temp dirs).

| Test | What it covers |
|---|---|
| `test_gguf_from_line` | FROM uses relative filename with first quantization |
| `test_merged_from_line` | FROM uses HuggingFace repo ID |
| `test_system_block_verbatim` | System prompt appears verbatim with triple-quote wrapper |
| `test_template_with_system_separator` | Go template emits system block when `system_separator` is set |
| `test_template_without_system_separator` | Go template emits inline system when `system_separator` is None |
| `test_parameters_all_present` | temperature, top_p, num_predict, num_ctx all appear |
| `test_repeat_penalty_included` | repeat_penalty appears when `repetition_penalty` is not None |
| `test_repeat_penalty_omitted` | repeat_penalty absent when `repetition_penalty` is None |
| `test_stop_strings` | One PARAMETER stop line per stop_string |
| `test_no_stop_lines_when_empty` | No PARAMETER stop lines when `stop_strings` is empty |
| `test_lora_receives_no_modelfile` | `generate_modelfile()` is not called for LoRA slug (verify via call path inspection) |
| `test_modelfile_overwritten_on_resave` | Second call overwrites first Modelfile |
| `test_publish_generates_modelfile_if_missing` | `publish_model()` generates Modelfile if not already on disk |

## Complexity Tracking

No constitution violations. No complexity tracking entries required.
