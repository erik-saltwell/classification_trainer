# Tasks: GGUF HuggingFace Metadata Files

**Input**: Design documents from `specs/012-gguf-hf-metadata/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Unit tests included — spec SC-005 requires automated coverage for non-GGUF format exclusion; SC-004 requires template consistency to be verifiable.

**Organization**: Tasks grouped by user story. All changes are in `helpers/publishing_helper.py` and one new test file.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, US3)
- Paths use single-project layout: `src/`, `tests/` at repo root

## Path Conventions

- Helpers: `src/classification_trainer/helpers/publishing_helper.py`
- Unit tests: `tests/unit/test_gguf_hf_metadata.py`

---

## Phase 1: Setup

**Purpose**: No new project structure or dependencies needed. One minimal change prepares the module.

- [X] T001 Add `import json` to `src/classification_trainer/helpers/publishing_helper.py` (at top of existing imports block)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Extract `_build_template_body()` from `generate_modelfile()`. This shared function is called by both `generate_modelfile()` and the new `generate_gguf_hf_metadata()`. Extracting it here prevents duplication and is a non-breaking refactor.

**⚠️ CRITICAL**: `generate_gguf_hf_metadata()` depends on `_build_template_body()` — no US1 implementation can begin until this phase is complete.

- [X] T00X Extract `_build_template_body(chat_template_info: ChatTemplateInfo) -> str` from the template-building block inside `generate_modelfile()` in `src/classification_trainer/helpers/publishing_helper.py`:
  - New private function placed just before `generate_modelfile()`
  - Body: compute `end_of_turn = chat_template_info.stop_strings[0] if chat_template_info.stop_strings else ""`; if `system_separator` is not None emit `f"{{{{- if .System }}}}{sys_sep}{{{{ .System }}}}{end_of_turn}\n{{{{- end }}}}\n"` else emit `"{{- if .System }}{{ .System }}\n{{- end }}\n"`; concatenate with prompt block `f"{{{{- if .Prompt }}}}{instr_sep}{{{{ .Prompt }}}}{end_of_turn}\n{{{{- end }}}}\n"` and `resp_sep`; return the assembled string
  - Update `generate_modelfile()` to replace its inline `system_part` / `template_body` computation with a call to `_build_template_body(chat_template_info)` — **no change to the generated Modelfile content**

**Checkpoint**: Run `pytest tests/unit/test_modelfile_generation.py -q` — all 17 existing Modelfile tests must still pass before proceeding.

---

## Phase 3: User Story 1 — Run GGUF Model in Ollama Without Local Download (Priority: P1) 🎯 MVP

**Goal**: When a GGUF model is saved, three files (`template`, `system`, `params`) are written to the save directory. They are automatically uploaded alongside the `.gguf` files and `Modelfile`, enabling `ollama run hf.co/user/repo` without any user configuration.

**Independent Test**: Save a GGUF model. Confirm `outputs/saves/<model>/gguf/template`, `/system`, and `/params` all exist. Verify `template` contains a Go template, `system` contains the verbatim system prompt, and `params` is valid JSON with the expected keys and values.

### Implementation for User Story 1

- [X] T00X [US1] Implement `generate_gguf_hf_metadata(save_dir: Path, training_info: TrainingInfo, publishing_info: PublishingInfo) -> None` in `src/classification_trainer/helpers/publishing_helper.py`:
  - Place after `generate_modelfile()` in the Modelfile generation section
  - `template`: `(save_dir / "template").write_text(_build_template_body(chat_template_info), encoding="utf-8")`
  - `system`: `(save_dir / "system").write_text(training_info.system_prompt, encoding="utf-8")`
  - `params`: build dict `{"temperature": inference_info.temperature, "top_p": inference_info.top_p, "num_predict": inference_info.max_new_tokens, "num_ctx": training_info.max_sequence_length, "stop": list(chat_template_info.stop_strings)}`; add `"repeat_penalty": inference_info.repetition_penalty` only if not None; write `json.dumps(params, indent=2)` to `save_dir / "params"`
- [X] T00X [US1] Wire `generate_gguf_hf_metadata()` into `_save_format()` inside `save_model()` in `src/classification_trainer/helpers/publishing_helper.py`:
  - Add immediately after the existing `generate_modelfile()` call (which is already guarded by `if slug in (SaveFormat.GGUF, SaveFormat.MERGED)`)
  - New guard: `if slug == SaveFormat.GGUF:` → log `f"    Generating HF metadata → {save_dir}/{{template,system,params}}"` → call `generate_gguf_hf_metadata(save_dir, training_info, publishing_info)`
  - This sits inside the existing `try/except` that calls `shutil.rmtree(save_dir)` on failure — no additional error handling needed
- [X] T00X [US1] Create `tests/unit/test_gguf_hf_metadata.py` with US1 unit tests (use `tmp_path` pytest fixture; reuse the `_training_info_mock`, `_inference`, `_chat_template`, `_publishing` helpers from `test_modelfile_generation.py` or copy them):
  - `test_template_file_written`: `template` file exists after call
  - `test_template_uses_system_separator`: when `system_separator` is set, template contains the separator string
  - `test_template_no_system_separator`: when `system_separator` is None, template uses inline system form
  - `test_system_file_verbatim`: `system` file equals `training_info.system_prompt` exactly
  - `test_params_required_keys`: JSON has `temperature`, `top_p`, `num_predict`, `num_ctx`, `stop`
  - `test_params_stop_array`: `stop` is a list with the correct stop string values
  - `test_params_valid_json`: `params` file parses without error via `json.loads()`
  - `test_all_three_files_written`: all three files exist after one call
  - `test_template_body_consistency`: `(save_dir / "template").read_text()` equals the `template_body` embedded in the co-located `Modelfile` TEMPLATE block — import `_build_template_body` and compare directly

**Checkpoint**: Run `pytest tests/unit/test_gguf_hf_metadata.py -q` — all 9 US1 tests must pass.

---

## Phase 4: User Story 2 — Pre-Existing GGUF Saves Get Metadata on Next Publish (Priority: P2)

**Goal**: A GGUF save directory created before this feature (no `template`/`system`/`params`) gets all three files generated automatically when `publish_model()` is called, before the folder is uploaded to HuggingFace.

**Independent Test**: Create a GGUF save directory with only `.gguf`, `README.md`, and `Modelfile`. Call `publish_model()` with a mocked `HfApi`. Confirm all three metadata files are created before `upload_folder` is called.

### Implementation for User Story 2

- [X] T00X [US2] Add GGUF metadata regeneration guard to `publish_model()` in `src/classification_trainer/helpers/publishing_helper.py`:
  - After the existing Modelfile regeneration block (which checks `not (save_dir / "Modelfile").exists()`), add:
    ```python
    if slug == SaveFormat.GGUF and not all(
        (save_dir / f).exists() for f in ("template", "system", "params")
    ):
        logger.report_message(f"    Generating missing HF metadata → {save_dir}/")
        generate_gguf_hf_metadata(save_dir, training_info, publishing_info)
    ```
  - This sits inside the existing `try/except` — if generation fails, the exception propagates into the `failures` list
- [X] T00X [US2] Add test `test_publish_regenerates_missing_hf_metadata` to `tests/unit/test_gguf_hf_metadata.py`:
  - Create a `tmp_path` GGUF directory with only `README.md` and `Modelfile` (no `template`/`system`/`params`)
  - Patch `CommonPaths` and `HfApi` (same pattern as `test_publish_generates_modelfile_if_missing` in `test_modelfile_generation.py`)
  - Call `publish_model()` and assert all three files exist afterward; assert `upload_folder` was called

**Checkpoint**: Run `pytest tests/unit/test_gguf_hf_metadata.py -q` — all 10 tests must pass.

---

## Phase 5: User Story 3 — Metadata Files Not Generated for Non-GGUF Formats (Priority: P3)

**Goal**: LoRA and merged save directories contain no `template`, `system`, or `params` files. The GGUF-only guard in `_save_format()` (added in T004) enforces this naturally; this phase verifies and tests the boundary.

**Independent Test**: Confirm merged save directory has no metadata files after a merged save. Verify via automated test.

### Implementation for User Story 3

- [X] T00X [US3] Verify and document the GGUF-only guard in `_save_format()` in `src/classification_trainer/helpers/publishing_helper.py`: confirm the condition is `if slug == SaveFormat.GGUF:` (not a broader condition). Add a brief inline comment: `# HF Ollama metadata files are GGUF-only; not applicable to merged or lora`
- [X] T00X [US3] Add test `test_not_generated_for_merged` to `tests/unit/test_gguf_hf_metadata.py`:
  - Call `generate_gguf_hf_metadata()` is NOT invoked for merged slug — verify by calling it directly with `SaveFormat.GGUF` (confirms it writes files) and separately confirming a `tmp_path` "merged dir" has no such files when the calling code uses `SaveFormat.MERGED` guard
  - Simplest form: create a merged-simulated `tmp_path`, assert `(merged_dir / "template").exists()` is False, `(merged_dir / "system").exists()` is False, `(merged_dir / "params").exists()` is False

**Checkpoint**: Run `pytest tests/unit/test_gguf_hf_metadata.py -q` — all 11 tests must pass. All 3 user stories independently verifiable.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Edge case coverage and full regression validation.

- [X] T01X [P] Add remaining edge case tests to `tests/unit/test_gguf_hf_metadata.py`:
  - `test_params_repeat_penalty_included`: `repeat_penalty` key present in parsed JSON when `InferenceInfo.repetition_penalty` is not None
  - `test_params_repeat_penalty_omitted`: `repeat_penalty` key absent from parsed JSON when `InferenceInfo.repetition_penalty` is None
  - `test_params_stop_empty_array`: when `stop_strings` is `()`, `stop` key in JSON is `[]`
  - `test_files_overwritten_on_resave`: second call to `generate_gguf_hf_metadata()` overwrites all three files with new content
- [X] T01X Run `pytest tests/unit/ -q` and confirm zero regressions across all 208+ pre-existing tests

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **BLOCKS US1 implementation**
- **Phase 3 (US1)**: Depends on Phase 2 complete (T002 must exist before T003)
- **Phase 4 (US2)**: Depends on Phase 2 + T003 (`generate_gguf_hf_metadata` function)
- **Phase 5 (US3)**: Depends on T004 (GGUF-only guard must be in place to verify)
- **Phase 6 (Polish)**: Depends on all story phases complete

### User Story Dependencies

- **US1 (P1)**: Depends on Phase 2 only
- **US2 (P2)**: Depends on Phase 2 + T003 (function) + T004 (call site)
- **US3 (P3)**: Depends on T004 (guard in `_save_format()`)

### Within Each Story

- T003 (function) before T004 (call site) before T005 (tests that call the real function)
- T006 (publish guard) before T007 (test for publish behavior)

### Parallel Opportunities

- T005 (US1 tests) and T006 (US2 publish guard) can be written in parallel — different files/concerns
- T010 edge case tests can be written in parallel with T011 regression check

---

## Parallel Example: Phase 3 + Phase 4

```bash
# After T003 + T004 complete:
# These can proceed in parallel:
Task T005: Write US1 tests in tests/unit/test_gguf_hf_metadata.py
Task T006: Wire publish_model() regeneration in helpers/publishing_helper.py
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: T001
2. Complete Phase 2: T002 (refactor + checkpoint)
3. Complete Phase 3: T003 → T004 → T005
4. **STOP and VALIDATE**: all 9 US1 tests pass; inspect generated files on disk
5. The three metadata files are now published with every GGUF save — core value delivered

### Incremental Delivery

1. Phase 1 + Phase 2 → `_build_template_body()` extracted, `json` imported
2. Phase 3 (US1) → files generated during save ← **ship here if needed**
3. Phase 4 (US2) → backward-compatible: old saves get files on next publish
4. Phase 5 (US3) → format boundary enforced and tested
5. Phase 6 → edge case coverage complete

---

## Notes

- [P] tasks = different files, no dependencies on each other
- All tests use `tmp_path` pytest fixture; no GPU or HuggingFace network calls needed
- The T002 refactor (`_build_template_body` extraction) must produce **zero change** to Modelfile output — the 17 existing `test_modelfile_generation.py` tests validate this
- The `publish_model()` guard (T006) uses `not all(...)` to trigger regeneration if *any* of the three files is missing, ensuring the set is always complete
- Commit after each phase checkpoint
