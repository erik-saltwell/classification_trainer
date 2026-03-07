# Tasks: Dataset Reference in Training Config

**Input**: Design documents from `/specs/006-dataset-ref-training-info/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Tests are included — the `_MINIMAL` dict in existing tests needs updating, and the new field needs validation tests.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Foundational (Blocking Prerequisites)

**Purpose**: Add the `dataset` field to `TrainingInfo` and update test fixtures — MUST be done before any user story work

- [x] T001 Add `dataset: str` field to `TrainingInfo` in `src/classification_trainer/configuration/training_info.py`, placed in the "References to reusable config files" section alongside `base_model`, `inference`, and `publishing`. Add a `dataset_info` property (following the `base_model_info`/`inference_info`/`publishing_info` pattern) that imports and calls `load_dataset_info(self.dataset)` to return a `DatasetInfo` instance.
- [x] T002 Update the `_MINIMAL` dict in `tests/unit/test_sft_parameters.py` to include `"dataset": "imdb"` so existing `TrainingInfo` tests continue to pass with the new required field.
- [x] T003 Write a unit test in `tests/unit/test_sft_parameters.py` (in the TrainingInfo integration section) verifying that `TrainingInfo` without a `dataset` field raises `ValidationError`.

**Checkpoint**: `TrainingInfo` requires a `dataset` field. All existing tests pass with updated fixtures.

---

## Phase 2: User Story 1 - Launch Commands with Only --training-info (Priority: P1)

**Goal**: Remove `--dataset` from all CLI commands and resolve dataset from the training config's `dataset` field.

**Independent Test**: Run any command with only `--training-info` and confirm it loads the correct dataset from the training config.

### Implementation for User Story 1

- [x] T004 [US1] Remove the `dataset_info: DatasetInfo` field from `AnalyzeDatasetCommand` in `src/classification_trainer/commands/analyze_dataset.py`. Update all references to `self.dataset_info` to use `self.training_info.dataset_info` instead.
- [x] T005 [US1] Remove the `dataset_info: DatasetInfo` field from `SweepCommand` in `src/classification_trainer/commands/sweep.py`. Update all references to `self.dataset_info` to use `self.training_info.dataset_info` instead.
- [x] T006 [US1] Remove the `dataset_info: DatasetInfo` field from `TrainCommand` in `src/classification_trainer/commands/train.py`. Update all references to `self.dataset_info` (including the local variable assignment) to use `self.training_info.dataset_info` instead.
- [x] T007 [US1] Remove the `dataset_info: DatasetInfo` field from `ComputeBatchSizeCommand` in `src/classification_trainer/commands/compute_batch_size.py`. Update all references to `self.dataset_info` to use `self.training_info.dataset_info` instead.
- [x] T008 [US1] Update all four command functions in `src/classification_trainer/console/main.py`: (a) remove the `dataset_info` parameter and `--dataset` option from `analyze_sequence_length`, `train`, `sweep`, and `compute_batch_size`; (b) remove the `ds_info = load_config_or_exit(load_dataset_info, ...)` call from each; (c) remove `dataset_info=ds_info` from the command constructor call in each; (d) remove the `load_dataset_info` import if no longer used.

**Checkpoint**: All four CLI commands work with only `--training-info`. The `--dataset` argument is gone from all commands.

---

## Phase 3: User Story 2 - Error on Missing/Invalid Dataset Reference (Priority: P2)

**Goal**: Clear errors when dataset field is missing or references a nonexistent file.

**Independent Test**: Run a command with a training config missing the `dataset` field, or referencing a nonexistent dataset name, and confirm clear errors.

### Implementation for User Story 2

No implementation tasks needed — the `dataset` field is required on `TrainingInfo` (Pydantic rejects missing fields), and `load_dataset_info` already raises `FileNotFoundError` with the expected path. Both behaviors are covered by T001 and T003.

**Checkpoint**: Missing field → Pydantic `ValidationError`. Nonexistent file → `FileNotFoundError` with path.

---

## Phase 4: User Story 3 - Documentation (Priority: P3)

**Goal**: Training config example YAML documents the `dataset` field.

**Independent Test**: Read `training_info/example.yaml` and confirm the `dataset` field is documented.

### Implementation for User Story 3

- [x] T009 [P] [US3] Add the `dataset` field to the "References to reusable config files" section of `training_info/example.yaml`, with inline comments following the same pattern as `base_model`, `inference`, and `publishing`. Example: `dataset: "imdb"` with comments explaining it's the filename stem of a file in `dataset_info/`.

**Checkpoint**: Example YAML documents the `dataset` field alongside existing reference fields.

---

## Phase 5: Migration & Polish

**Purpose**: Migrate existing configs and run final validation

- [x] T010 [P] Add `dataset: "imdb"` to `training_info/imdb.yaml`.
- [x] T011 [P] Add `dataset: "reddit-rpg-questions"` to `training_info/reddit-rpg-questions-classifier.yaml`.
- [x] T012 [P] Add `dataset: "test-reddit-questions"` to `training_info/test-reddit-questions.yaml`.
- [x] T013 Run all tests with `cd src && pytest` to verify no regressions across all existing test files.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 1)**: No dependencies — can start immediately. BLOCKS all user stories.
- **US1 (Phase 2)**: Depends on Phase 1 (needs `dataset` field and `dataset_info` property on TrainingInfo)
- **US2 (Phase 3)**: No implementation needed — covered by Phase 1
- **US3 (Phase 4)**: Independent of US1 — can run in parallel after Phase 1
- **Migration (Phase 5)**: Depends on Phase 1 (configs need the new field). T010-T012 can run in parallel with US1.
- **T013**: Depends on ALL phases being complete

### Parallel Opportunities

- T004, T005, T006, T007 can run in parallel (different command files)
- T009 can run in parallel with US1 tasks (different file)
- T010, T011, T012 can run in parallel (different config files)

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Foundational (T001, T002, T003)
2. Complete Phase 2: User Story 1 (T004–T008)
3. **STOP and VALIDATE**: Run commands with only `--training-info`

### Incremental Delivery

1. Foundational → Model and property ready
2. US1 → CLI simplified → **MVP**
3. US3 → Documentation in example.yaml
4. Migration → Existing configs updated
5. Polish → Full test run

---

## Notes

- T004–T007 modify different command files and can run in parallel
- T008 modifies main.py and depends on T004–T007 being conceptually clear (but can be done together since the changes are independent within main.py)
- US2 requires no implementation — Pydantic and existing loader handle errors automatically
- The `_MINIMAL` dict in test_sft_parameters.py is the only test fixture that needs updating
