# Implementation Plan: GGUF HuggingFace Metadata Files

**Branch**: `012-gguf-hf-metadata` | **Date**: 2026-03-17 | **Spec**: [spec.md](spec.md)
**Input**: When publishing a GGUF model, generate and upload `template`, `system`, and `params` files alongside the existing `Modelfile` so that `ollama run hf.co/<user>/<repo>` works without any local download.

## Summary

Add three HuggingFace Ollama metadata files (`template`, `system`, `params`) to the GGUF save directory during save and publish. These files are consumed by Ollama when a user runs `ollama run hf.co/user/repo` directly — Ollama downloads the GGUF and reads these root-level files to configure the model without requiring a local `Modelfile`. The implementation refactors the template-body building logic out of `generate_modelfile()` into a shared private helper and adds a new `generate_gguf_hf_metadata()` function. No new config models, YAML files, or dependencies are required (beyond stdlib `json`).

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: stdlib `json` only (no new packages)
**Storage**: Three plain-text files written to existing GGUF save directory
**Testing**: pytest unit tests; no GPU or HuggingFace network calls required
**Target Platform**: Linux (same as existing project)
**Project Type**: CLI tool / library
**Performance Goals**: N/A — file generation is negligible overhead
**Constraints**: GGUF format only; additive to existing Modelfile; no new config flags
**Scale/Scope**: Three additional files per GGUF publish

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Configuration-First | ✅ PASS | All file content derived from existing Pydantic models. No hardcoded values. No new YAML or config models needed. |
| II. Protocol-Based Interfaces | ✅ PASS | `generate_gguf_hf_metadata()` logs via `LoggingProtocol`. No new protocols needed. |
| III. Separation of Concerns | ✅ PASS | New function in `helpers/publishing_helper.py`. Commands unchanged. |
| IV. Observability | ✅ PASS | Each file generation logged via `LoggingProtocol`. Failure propagates via existing cleanup. |
| V. Simplicity & Scope | ✅ PASS | Additive to an existing publish feature. `_build_template_body()` extraction is real deduplication, not premature abstraction. |

## Project Structure

### Documentation (this feature)

```text
specs/012-gguf-hf-metadata/
├── plan.md                         # This file
├── research.md                     # HF/Ollama metadata file format research
├── data-model.md                   # Code entities: _build_template_body, generate_gguf_hf_metadata
├── contracts/
│   └── hf-metadata-files.md        # template / system / params format contract
└── tasks.md                        # Phase 2 output (/speckit.tasks)
```

### Source Code Changes

```text
# Modified files
src/classification_trainer/helpers/publishing_helper.py
    - Add import json
    - Extract _build_template_body(chat_template_info) -> str from generate_modelfile()
    - Update generate_modelfile() to call _build_template_body()
    - Add generate_gguf_hf_metadata(save_dir, training_info, publishing_info) -> None
    - In _save_format(): call generate_gguf_hf_metadata() after generate_modelfile() for GGUF only
    - In publish_model(): regenerate missing HF metadata files for GGUF (mirrors Modelfile logic)

# New test file
tests/unit/test_gguf_hf_metadata.py
    - Unit tests for generate_gguf_hf_metadata() and _build_template_body() consistency

# No YAML or config model changes required
```

**Structure Decision**: Single-project layout. All changes stay within `helpers/publishing_helper.py` and one new test file.

## Phase 0: Research

*Complete. See [research.md](research.md).*

Key findings:
- Three files at repo root: `template` (Go template body), `system` (plain text), `params` (JSON)
- GGUF-only: merged repos use the Modelfile approach instead
- Template body content is identical to what `generate_modelfile()` already computes — extract `_build_template_body()` to share
- `params` JSON uses the same parameter names as Modelfile PARAMETER keys
- stdlib `json` module is sufficient; no new packages needed

## Phase 1: Design

*See [data-model.md](data-model.md) and [contracts/hf-metadata-files.md](contracts/hf-metadata-files.md).*

### Refactor: `_build_template_body(chat_template_info)`

Extract from the existing `generate_modelfile()` template-building block into a standalone private function. `generate_modelfile()` is updated to call it — **no behavioral change to Modelfile output**.

```python
def _build_template_body(chat_template_info: ChatTemplateInfo) -> str:
    end_of_turn = chat_template_info.stop_strings[0] if chat_template_info.stop_strings else ""
    instr_sep = chat_template_info.instruction_separator
    resp_sep = chat_template_info.response_separator
    sys_sep = chat_template_info.system_separator

    if sys_sep is not None:
        system_part = f"{{{{- if .System }}}}{sys_sep}{{{{ .System }}}}{end_of_turn}\n{{{{- end }}}}\n"
    else:
        system_part = "{{- if .System }}{{ .System }}\n{{- end }}\n"

    return (
        system_part
        + f"{{{{- if .Prompt }}}}{instr_sep}{{{{ .Prompt }}}}{end_of_turn}\n{{{{- end }}}}\n"
        + resp_sep
    )
```

### New function: `generate_gguf_hf_metadata(save_dir, training_info, publishing_info)`

Writes `template`, `system`, and `params` to `save_dir`:

```python
def generate_gguf_hf_metadata(
    save_dir: Path,
    training_info: TrainingInfo,
    publishing_info: PublishingInfo,
) -> None:
    chat_template_info = training_info.base_model_info.chat_template_info
    inference_info = training_info.inference_info

    (save_dir / "template").write_text(_build_template_body(chat_template_info), encoding="utf-8")
    (save_dir / "system").write_text(training_info.system_prompt, encoding="utf-8")

    params: dict = {
        "temperature": inference_info.temperature,
        "top_p": inference_info.top_p,
        "num_predict": inference_info.max_new_tokens,
        "num_ctx": training_info.max_sequence_length,
        "stop": list(chat_template_info.stop_strings),
    }
    if inference_info.repetition_penalty is not None:
        params["repeat_penalty"] = inference_info.repetition_penalty
    (save_dir / "params").write_text(json.dumps(params, indent=2), encoding="utf-8")
```

### Call site in `_save_format()`

After the existing `generate_modelfile()` call:

```python
if slug == SaveFormat.GGUF:
    logger.report_message(f"    Generating HF metadata → {save_dir}/{{template,system,params}}")
    generate_gguf_hf_metadata(save_dir, training_info, publishing_info)
```

Inside the existing `try/except` — failure cleans up the save directory and re-raises.

### Call site in `publish_model()`

After the existing Modelfile regeneration block:

```python
if slug == SaveFormat.GGUF and not all(
    (save_dir / f).exists() for f in ("template", "system", "params")
):
    logger.report_message(f"    Generating missing HF metadata → {save_dir}/")
    generate_gguf_hf_metadata(save_dir, training_info, publishing_info)
```

### Test Design

**File**: `tests/unit/test_gguf_hf_metadata.py`

| Test | Coverage |
|---|---|
| `test_template_file_written` | `template` file exists after call |
| `test_template_uses_system_separator` | system block in template when `system_separator` set |
| `test_template_no_system_separator` | inline system block when `system_separator` is None |
| `test_system_file_verbatim` | `system` file = verbatim system prompt |
| `test_params_required_keys` | temperature, top_p, num_predict, num_ctx, stop all present |
| `test_params_stop_array` | stop is a JSON array with correct entries |
| `test_params_repeat_penalty_included` | repeat_penalty present when not None |
| `test_params_repeat_penalty_omitted` | repeat_penalty absent when None |
| `test_params_valid_json` | params file parses as valid JSON |
| `test_all_three_files_written` | all three files exist after call |
| `test_files_overwritten_on_resave` | second call overwrites all three |
| `test_template_body_consistency` | `template` file body matches Modelfile TEMPLATE body |
| `test_publish_regenerates_missing_hf_metadata` | `publish_model()` generates files if any is missing |
| `test_not_generated_for_merged` | metadata files absent when only merged format called |

## Complexity Tracking

No constitution violations. No complexity tracking entries required.
