# Tasks: Modelfile Generation on Publish

**Input**: Design documents from `specs/011-generate-modelfile/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Unit tests are included (spec SC-004 requires automated test coverage for LoRA exclusion; SC-003 requires parameter derivation to be verifiable).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths use single-project layout: `src/`, `tests/` at repo root

## Path Conventions

- Config models: `src/classification_trainer/configuration/`
- Helpers: `src/classification_trainer/helpers/`
- Chat template YAMLs: `chat_template_info/`
- Unit tests: `tests/unit/`

---

## Phase 1: Setup

**Purpose**: No new project structure or dependencies needed. One documentation update to the example config file.

- [x] T001 Add `system_separator` field documentation to `chat_template_info/example.yaml` with full inline comments explaining its purpose and accepted values

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add `system_separator: str | None = None` to `ChatTemplateInfo` and populate all existing YAML files. Every user story depends on this field being available.

**⚠️ CRITICAL**: `generate_modelfile()` reads `chat_template_info.system_separator` — no user story implementation can begin until this phase is complete.

- [x] T002 Add `system_separator: str | None = None` field to `ChatTemplateInfo` in `src/classification_trainer/configuration/chat_template_info.py` (after `assistant_newline` field; default None preserves backward compatibility)
- [x] T003 [P] Add `system_separator: "<|im_start|>system\n"` to `chat_template_info/chat-ml.yaml`
- [x] T004 [P] Add `system_separator: "<|start_header_id|>system<|end_header_id|>\n\n"` to `chat_template_info/llama.yaml`
- [x] T005 [P] Add `system_separator: null` to `chat_template_info/mistral.yaml` (Mistral renders system inline before `[INST]`)
- [x] T006 [P] Add `system_separator: "<start_of_turn>system\n"` to `chat_template_info/gemma.yaml`
- [x] T007 [P] Add `system_separator: "<|system|>\n"` to `chat_template_info/phi.yaml`

**Checkpoint**: Run `pytest tests/unit/test_training_info.py` — all existing tests must still pass before proceeding.

---

## Phase 3: User Story 1 — GGUF Model Ready for Ollama Out of the Box (Priority: P1) 🎯 MVP

**Goal**: When a GGUF model is saved, a `Modelfile` is written alongside the `.gguf` files. The file contains a correct `FROM` relative filename, verbatim `SYSTEM` prompt, Go-template `TEMPLATE` block, and all inference `PARAMETER` lines derived from config.

**Independent Test**: Save a GGUF model using the existing `save_model()` code path. Confirm `outputs/saves/<model>/gguf/Modelfile` exists. Open it and verify the `FROM` line equals `<model_name>-gguf-<first_quant>.gguf`, the `SYSTEM` block contains the system prompt verbatim, the `TEMPLATE` block uses the correct separators, and all PARAMETER values match `InferenceInfo` and `TrainingInfo`.

### Implementation for User Story 1

- [x] T008 [US1] Implement `generate_modelfile(save_dir, format_slug, training_info, publishing_info)` in `src/classification_trainer/helpers/publishing_helper.py`:
  - Build `FROM` line: GGUF → `f"FROM {training_info.model_name}-gguf-{publishing_info.gguf_quantizations[0]}.gguf"`; merged → `f"FROM {training_info.hugging_face_user_name}/{training_info.model_name}-merged"` (both branches in one function)
  - Build `SYSTEM` block: `f'SYSTEM """\n{training_info.system_prompt}\n"""'`
  - Build `TEMPLATE` block from `training_info.base_model_info.chat_template_info`: derive `end_of_turn = stop_strings[0]` if stop_strings else `""`; if `system_separator` is not None emit `{{ if .System }}{system_separator}{{ .System }}{end_of_turn}\n{{ end }}`; else emit `{{ if .System }}{{ .System }}\n{{ end }}`; always emit `{{ if .Prompt }}{instruction_separator}{{ .Prompt }}{end_of_turn}\n{{ end }}{response_separator}`; wrap in `TEMPLATE """\n...\n"""`
  - Build PARAMETER lines: always `temperature`, `top_p`, `num_predict` (from `inference_info.max_new_tokens`), `num_ctx` (from `training_info.max_sequence_length`); conditionally `repeat_penalty` if `inference_info.repetition_penalty is not None`; one `PARAMETER stop "..."` per entry in `stop_strings`
  - Assemble with `"\n\n".join(...)` and write to `save_dir / "Modelfile"` via `Path.write_text(content, encoding="utf-8")`
- [x] T009 [US1] Wire `generate_modelfile()` call inside `_save_format()` in `src/classification_trainer/helpers/publishing_helper.py`: add call immediately after `generate_model_card(...)` for `SaveFormat.GGUF` only (use `if slug == SaveFormat.GGUF:` guard); call sits inside the existing `try/except` that does `shutil.rmtree(save_dir)` on failure — no extra error handling needed
- [x] T010 [US1] Create `tests/unit/test_modelfile_generation.py` with unit tests for GGUF behavior using in-memory minimal config objects (no GPU, no HuggingFace, use `tmp_path` pytest fixture for file I/O):
  - `test_gguf_from_line`: FROM equals `<model_name>-gguf-<first_quant>.gguf`
  - `test_system_block_verbatim`: SYSTEM block contains verbatim system prompt wrapped in triple-quotes
  - `test_template_with_system_separator`: when `system_separator` is set, TEMPLATE emits `{{ if .System }}<separator>{{ .System }}...` system block
  - `test_template_without_system_separator`: when `system_separator` is None, TEMPLATE emits inline `{{ if .System }}{{ .System }}\n{{ end }}`
  - `test_parameters_all_present`: temperature, top_p, num_predict, num_ctx all appear in correct order
  - `test_stop_strings_in_parameters`: one `PARAMETER stop` line per stop_string entry
  - `test_modelfile_written_to_disk`: Modelfile is created at `save_dir / "Modelfile"` with UTF-8 encoding

**Checkpoint**: Run `pytest tests/unit/test_modelfile_generation.py` — all 7 GGUF tests must pass. Manually inspect a generated Modelfile to confirm Ollama can parse it (`ollama create test -f Modelfile`).

---

## Phase 4: User Story 2 — Merged Model Published with Modelfile (Priority: P2)

**Goal**: When a merged model is saved, a `Modelfile` is written with `FROM <hf-user>/<model>-merged`. When `publish_model()` is called on a qualifying format directory that has no `Modelfile`, it generates one before uploading.

**Independent Test**: Save a merged model. Confirm `outputs/saves/<model>/merged/Modelfile` exists with `FROM username/model-merged` as the first line. Separately, delete the Modelfile from a GGUF save directory and call `publish_model()` (mocked HfApi); confirm the Modelfile is regenerated before `api.upload_folder()` is called.

### Implementation for User Story 2

- [x] T011 [US2] Wire `generate_modelfile()` call inside `_save_format()` in `src/classification_trainer/helpers/publishing_helper.py` for `SaveFormat.MERGED`: extend the existing GGUF guard to cover merged — change `if slug == SaveFormat.GGUF:` to `if slug in (SaveFormat.GGUF, SaveFormat.MERGED):` (or add an `elif slug == SaveFormat.MERGED:` block — same call, same position relative to `generate_model_card`)
- [x] T012 [US2] Add missing-Modelfile regeneration to `publish_model()` in `src/classification_trainer/helpers/publishing_helper.py`: after the `readme` existence check and before `api.upload_folder()`, add: `if slug in (SaveFormat.GGUF, SaveFormat.MERGED) and not (save_dir / "Modelfile").exists(): generate_modelfile(save_dir, slug, training_info, publishing_info)` — if generation raises, let the existing `except Exception as exc` catch it and append slug to `failures`
- [x] T013 [US2] Extend `tests/unit/test_modelfile_generation.py` with merged and publish tests:
  - `test_merged_from_line`: FROM equals `<hf_user>/<model_name>-merged`
  - `test_repeat_penalty_included`: `PARAMETER repeat_penalty` line appears when `InferenceInfo.repetition_penalty` is not None
  - `test_repeat_penalty_omitted`: no `PARAMETER repeat_penalty` line when `InferenceInfo.repetition_penalty` is None
  - `test_no_stop_lines_when_empty`: no `PARAMETER stop` lines when `stop_strings` is `()`
  - `test_publish_generates_modelfile_if_missing`: with a save directory that has no `Modelfile`, calling `publish_model()` (with mocked `HfApi`) results in a `Modelfile` being created before `upload_folder` is called

**Checkpoint**: Run `pytest tests/unit/test_modelfile_generation.py` — all 12 tests must pass.

---

## Phase 5: User Story 3 — LoRA Adapter Excluded from Modelfile Generation (Priority: P3)

**Goal**: `generate_modelfile()` is never called for `SaveFormat.LORA`. No `Modelfile` appears in the LoRA save directory after save or publish.

**Independent Test**: Save only a LoRA adapter (set `save_lora=True`, `save_gguf=False`, `save_merged=False`). Confirm `outputs/saves/<model>/lora/` contains no `Modelfile`. Also verify the guard in `publish_model()` skips LoRA when checking for missing Modelfiles.

### Implementation for User Story 3

- [x] T014 [US3] Verify (and document via comment) that the guard added in T009/T011 in `_save_format()` in `src/classification_trainer/helpers/publishing_helper.py` naturally excludes LoRA — the condition `slug in (SaveFormat.GGUF, SaveFormat.MERGED)` means `SaveFormat.LORA` never triggers `generate_modelfile()`. If the condition was written as a simple `if slug == SaveFormat.GGUF` in T009, update it now to the combined form covering both qualifying formats.
- [x] T015 [US3] Add unit test `test_lora_receives_no_modelfile` to `tests/unit/test_modelfile_generation.py`: create a `tmp_path` save directory simulating a LoRA save (no Modelfile written), confirm `(save_dir / "Modelfile").exists()` is False after running the relevant code path with `SaveFormat.LORA`

**Checkpoint**: Run `pytest tests/unit/test_modelfile_generation.py` — all 13 tests must pass. All 3 user stories are independently verifiable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Edge case coverage, overwrite behavior, and final validation.

- [x] T016 [P] Add remaining edge case tests to `tests/unit/test_modelfile_generation.py`:
  - `test_modelfile_overwritten_on_resave`: calling `generate_modelfile()` twice on the same directory produces a single `Modelfile` with the second call's content (no duplicate or stale content)
  - `test_multiple_stop_strings`: GGUF Modelfile with two stop strings produces two `PARAMETER stop` lines
  - `test_multiple_quantizations_uses_first`: with `gguf_quantizations = ["q4_k_m", "q8_0"]`, FROM references `q4_k_m` file
  - `test_system_prompt_with_newlines`: system prompt containing embedded newlines is preserved verbatim inside triple-quote block
- [x] T017 Run `pytest tests/unit/` and confirm no regressions in pre-existing tests (`test_training_info.py`, `test_cli.py`, etc.)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — can start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **BLOCKS all user stories**
- **Phase 3 (US1 — GGUF)**: Depends on Phase 2 complete
- **Phase 4 (US2 — Merged)**: Depends on Phase 2 complete; T011/T012 depend on T008 (generate_modelfile exists)
- **Phase 5 (US3 — LoRA exclusion)**: Depends on Phase 2; T014 depends on T009/T011 guard being in place
- **Phase 6 (Polish)**: Depends on all story phases complete

### User Story Dependencies

- **US1 (P1)**: Depends on Phase 2 only — no other story dependency
- **US2 (P2)**: Depends on Phase 2 + T008 (generate_modelfile function from US1)
- **US3 (P3)**: Depends on Phase 2 + T009/T011 (call sites from US1/US2)

### Within Each Story

- Config model change (Phase 2) before helper implementation (US1)
- `generate_modelfile()` function (T008) before call sites (T009, T011, T012)
- Implementation before tests that run the real function

### Parallel Opportunities

- T003–T007 (YAML file updates) are all independent and can run simultaneously
- T010 (US1 tests) can be written before T008 (TDD: write tests, confirm they fail, then implement)
- T016 edge case tests can be written in parallel with T017 regression check

---

## Parallel Example: Phase 2 (Foundational)

```bash
# All YAML updates are independent — run simultaneously:
Task: T003 — Add system_separator to chat_template_info/chat-ml.yaml
Task: T004 — Add system_separator to chat_template_info/llama.yaml
Task: T005 — Add system_separator to chat_template_info/mistral.yaml
Task: T006 — Add system_separator to chat_template_info/gemma.yaml
Task: T007 — Add system_separator to chat_template_info/phi.yaml
# T002 must complete first (adds field to ChatTemplateInfo Pydantic model)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: T001
2. Complete Phase 2: T002 → T003–T007 (parallel)
3. Complete Phase 3: T008 → T009 → T010
4. **STOP and VALIDATE**: Run tests, inspect a real Modelfile from a GGUF save
5. US1 delivers the highest-value use case (GGUF is Ollama's native format)

### Incremental Delivery

1. Phase 1 + Phase 2 → ChatTemplateInfo field in place
2. Phase 3 (US1) → GGUF Modelfile works end-to-end ← **ship here if needed**
3. Phase 4 (US2) → Merged Modelfile + publish regeneration
4. Phase 5 (US3) → LoRA exclusion enforced and tested
5. Phase 6 → Full edge case coverage

---

## Notes

- [P] tasks = different files, no dependencies on each other
- All tests use pytest with `tmp_path` fixture; no GPU or HuggingFace network calls needed
- `generate_modelfile()` failure propagates into the existing `shutil.rmtree` cleanup — no new error handling needed
- The YAML changes (T003–T007) add a new optional field; existing tests continue to pass because `system_separator` defaults to `None`
- Commit after each phase checkpoint to keep changes reviewable
