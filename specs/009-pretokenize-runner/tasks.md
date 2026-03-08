# Tasks: Pre-tokenize Runner

**Input**: Design documents from `/specs/009-pretokenize-runner/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md

**Tests**: Included — the plan explicitly specifies test file modifications.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Foundational (Blocking Prerequisites)

**Purpose**: Add the `pretokenize` parameter and thread it through the call chain. Register new column names for cleanup.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T001 Add `pretokenize: bool = False` field to the `TrainingRunner` dataclass in `src/classification_trainer/commands/training_runner.py`. Pass it through `prepare_data()` to `prepare_split_data()`.
- [X] T002 Add `pretokenize: bool = False` parameter to `prepare_split_data()` and `prep_dataset()` in `src/classification_trainer/helpers/dataset_helper.py`. Thread it from `prepare_split_data` into each `prep_dataset` call. When `False`, behavior is unchanged.
- [X] T003 [P] Add `input_ids`, `attention_mask`, `labels`, `eval_input_ids`, `eval_attention_mask` to `get_generated_column_names()` in `src/classification_trainer/configuration/dataset_info.py` (FR-008).
- [X] T004 [P] Add test in `tests/unit/test_dataset_info.py` verifying the five new column names appear in `get_generated_column_names()`.

**Checkpoint**: `pretokenize` parameter flows from TrainingRunner through to prep_dataset. When `False`, all behavior identical. Existing tests pass (`cd src && pytest`).

---

## Phase 2: User Story 2 - Pre-tokenize Training Data (Priority: P1) 🎯 MVP

**Goal**: When `pretokenize=True`, `prepare_data` adds `input_ids`, `attention_mask`, and (conditionally) `labels` columns. The trainer auto-detects these and skips re-tokenization.

**Independent Test**: Create a TrainingRunner with `pretokenize=True`, call `prepare_data`, verify token columns exist in the dataset. Verify `create_trainer()` sets `dataset_text_field=None` and skips `train_on_responses_only()` when `input_ids` is present.

### Implementation for User Story 2

- [X] T005 [US2] Add `tokenize_training_column(dataset_info, dataset, tokenizer, max_seq_len) -> Dataset` function in `src/classification_trainer/helpers/dataset_helper.py`. Tokenizes the training text column via `dataset.map`, adding `input_ids` and `attention_mask` columns. Uses `truncation=True, max_length=max_seq_len, add_special_tokens=False`.
- [X] T006 [US2] Add `_find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int | None` helper in `src/classification_trainer/helpers/dataset_helper.py`. Searches backwards through `sequence` for the last occurrence of `subsequence`, returns start index or `None`.
- [X] T007 [US2] Add `apply_response_masking(dataset, tokenizer, chat_template_info) -> Dataset` function in `src/classification_trainer/helpers/dataset_helper.py`. Tokenizes `chat_template_info.response_separator` to get separator token IDs, then for each row: finds last occurrence of separator in `input_ids` via `_find_last_subsequence`, sets `labels[0:sep_end] = -100`, keeps `labels[sep_end:] = input_ids[sep_end:]`. If separator not found, all labels = -100.
- [X] T008 [US2] Update `prep_dataset()` in `src/classification_trainer/helpers/dataset_helper.py` to call `tokenize_training_column()` and (if `training_info.train_on_outputs_only`) `apply_response_masking()` when `pretokenize=True`.
- [X] T009 [US2] Add `pre_tokenized: bool = False` parameter to `create_sft_config()` in `src/classification_trainer/configuration/training_info.py`. When `True`, set `dataset_text_field=None` instead of `dataset_info.training_column_name` (FR-005). See research.md R1: SFTTrainer auto-detects `input_ids` and skips tokenization.
- [X] T010 [US2] Update `create_trainer()` in `src/classification_trainer/helpers/training_helper.py`: auto-detect `pre_tokenized = "input_ids" in train_dataset.column_names`, pass to `create_sft_config()`, and skip `train_on_responses_only()` when `pre_tokenized` is `True` (FR-006). See research.md R3: the function would overwrite pre-computed labels.

### Tests for User Story 2

- [X] T011 [P] [US2] Add tests for `tokenize_training_column` in `tests/unit/test_dataset_helper.py`: verify `input_ids` and `attention_mask` columns are added, verify truncation at max_seq_len, verify `add_special_tokens=False` is used.
- [X] T012 [P] [US2] Add tests for `_find_last_subsequence` in `tests/unit/test_dataset_helper.py`: found at end, found at start, found in middle, not found returns None, multiple occurrences returns last.
- [X] T013 [P] [US2] Add tests for `apply_response_masking` in `tests/unit/test_dataset_helper.py`: verify labels before separator are -100, labels after separator match input_ids, separator not found → all -100.
- [X] T014 [P] [US2] Add test for `create_sft_config(pre_tokenized=True)` in `tests/unit/test_training_info.py`: verify `dataset_text_field` is `None`. Add test for `create_sft_config(pre_tokenized=False)`: verify `dataset_text_field` equals `dataset_info.training_column_name`.

**Checkpoint**: With `pretokenize=True`, training datasets have `input_ids`/`attention_mask`/`labels` columns. `create_trainer()` detects them and configures SFTTrainer to skip tokenization. All tests pass (`cd src && pytest`).

---

## Phase 3: User Story 3 - Pre-tokenize Eval Data (Priority: P2)

**Goal**: When `pretokenize=True`, `prepare_data` adds `eval_input_ids` and `eval_attention_mask` columns. Inference auto-detects these and generates from tokens directly.

**Independent Test**: Create a TrainingRunner with `pretokenize=True`, call `prepare_data`, verify eval token columns exist. Verify `add_inferred_column` uses the token-based path when `eval_input_ids` is present.

### Implementation for User Story 3

- [X] T015 [US3] Add `_clean_prompt_ending(prompt_text, template) -> str` in `src/classification_trainer/helpers/dataset_helper.py`. Duplicate of the same function in `inference_helper.py` — ensures prompt ends with newline after response separator when `assistant_newline=True`. Duplicated to avoid importing heavy torch/unsloth dependencies (see plan.md D2).
- [X] T016 [US3] Add `tokenize_eval_column(dataset_info, dataset, tokenizer, chat_template_info) -> Dataset` in `src/classification_trainer/helpers/dataset_helper.py`. Applies `_clean_prompt_ending` to each eval prompt text, then tokenizes with `add_special_tokens=chat_template_info.add_special_tokens` and no truncation. Adds `eval_input_ids` and `eval_attention_mask` columns.
- [X] T017 [US3] Update `prep_dataset()` in `src/classification_trainer/helpers/dataset_helper.py` to call `tokenize_eval_column()` when `pretokenize=True`.
- [X] T018 [US3] Add `generate_label_text_from_tokens(model, tokenizer, input_ids, attention_mask, inference_info, template) -> str` in `src/classification_trainer/helpers/inference_helper.py`. Takes `input_ids` and `attention_mask` as `list[int]`, converts to tensors on device, builds generate kwargs via existing `_build_generate_kwargs`, calls `model.generate()`, decodes and trims via existing `_decode_and_trim_generated_texts` (FR-007).
- [X] T019 [US3] Update `add_inferred_column()` in `src/classification_trainer/helpers/inference_helper.py` to detect `"eval_input_ids" in dataset.column_names`. When present, iterate batch rows calling `generate_label_text_from_tokens()` with `eval_input_ids` and `eval_attention_mask` values. When absent, use the existing text-based path unchanged.

### Tests for User Story 3

- [X] T020 [P] [US3] Add tests for `tokenize_eval_column` in `tests/unit/test_dataset_helper.py`: verify `eval_input_ids` and `eval_attention_mask` columns are added, verify `_clean_prompt_ending` is applied before tokenization.

**Checkpoint**: With `pretokenize=True`, all dataset splits have eval token columns. Inference detects them and generates from tokens. All tests pass (`cd src && pytest`).

---

## Phase 4: Verify & Polish

**Purpose**: Validate backward compatibility (US1) and cross-cutting concerns

- [X] T021 Run full existing test suite (`cd src && pytest`) to verify all existing tests pass without modification (SC-001, US1).
- [X] T022 Run linter (`cd src && ruff check .`) and fix any issues in modified files.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 1)**: No dependencies — start immediately
- **US2 (Phase 2)**: Depends on Phase 1 (T001, T002)
- **US3 (Phase 3)**: Depends on Phase 1 (T001, T002) and Phase 2 (T008 — prep_dataset changes)
- **Verify & Polish (Phase 4)**: Depends on all previous phases

### User Story Dependencies

- **US2 (P1)**: Can start after Phase 1 — no dependency on other stories
- **US3 (P2)**: Depends on US2 completion (T008 updates prep_dataset which US3's T017 extends)
- **US4 (P3)**: Addressed in Phase 1 (T003) — no dependency on stories
- **US1 (P1)**: Verified in Phase 4 (T021) — depends on all changes being complete

### Within Each User Story

- Helper functions before callers (e.g., T005-T007 before T008)
- Config changes before helper changes that use them (T009 before T010)
- Implementation before tests (tests reference the functions)

### Parallel Opportunities

- T003 and T004 can run in parallel with T001/T002 (different files)
- T011, T012, T013, T014 can all run in parallel (different test functions, different files)
- T020 can run in parallel with other US3 tests

---

## Parallel Example: User Story 2

```bash
# After T005-T010 are complete, launch all US2 tests together:
Task T011: "Test tokenize_training_column in tests/unit/test_dataset_helper.py"
Task T012: "Test _find_last_subsequence in tests/unit/test_dataset_helper.py"
Task T013: "Test apply_response_masking in tests/unit/test_dataset_helper.py"
Task T014: "Test create_sft_config pre_tokenized in tests/unit/test_training_info.py"
```

---

## Implementation Strategy

### MVP First (User Story 2 Only)

1. Complete Phase 1: Foundational (T001-T004)
2. Complete Phase 2: User Story 2 (T005-T014)
3. **STOP and VALIDATE**: Run `cd src && pytest` — all existing + new tests pass
4. Training pre-tokenization is functional

### Incremental Delivery

1. Phase 1 → Parameter plumbing + column cleanup ready
2. Phase 2 (US2) → Training pre-tokenization works → Validate
3. Phase 3 (US3) → Eval pre-tokenization works → Validate
4. Phase 4 → Full regression check + lint

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US1 (backward compatibility) is validated by existing tests passing — not a separate implementation phase
- US4 (column cleanup) is a single foundational task (T003) since it's just adding yields
- All changes are modifications to existing files — no new files created
- Commit after each phase checkpoint
