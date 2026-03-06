# Tasks: Configuration File Reorganization

**Input**: Design documents from `/specs/001-reorganize-config/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/cli.md ✅

**Tests**: Test tasks are included for the Pydantic model changes and CLI contract (FR-004 requires clear error messages on missing referenced configs).

**Organization**: Tasks are grouped by user story. US2 (Discover and Browse Reusable Configs) is achieved entirely by US1's migration work — it has no unique implementation tasks.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on each other)
- **[Story]**: Which user story this task belongs to (US1, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup

**Purpose**: No new project structure needed — all directories already exist. One preparatory task.

- [x] T001 Read publishing_helper.py fully to confirm the exact function signature and call site that uses `publishing_info.description` before any changes are made: `src/classification_trainer/helpers/publishing_helper.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Pydantic model changes that ALL other phases depend on. Must complete before US1 work begins.

**⚠️ CRITICAL**: CLI, helper, and YAML migration tasks cannot be completed correctly until these models are finalized.

- [x] T002 Update `TrainingInfo` Pydantic model: add fields `base_model: str`, `inference: str`, `publishing: str | None = None`, `model_card_description: str`; add resolution properties `base_model_info`, `inference_info`, `publishing_info` using lazy imports inside property bodies (follow the `BaseModelInfo.chat_template_info` pattern) in `src/classification_trainer/configuration/training_info.py`
- [x] T003 [P] Remove `description: str` field from `PublishingInfo` Pydantic model in `src/classification_trainer/configuration/publishing_info.py`
- [x] T004 [P] Add `self.inference_info.mkdir(parents=True, exist_ok=True)` to `ensure_all_dirs_exist()` in `src/classification_trainer/utils/common_paths.py`

**Checkpoint**: Pydantic models updated — CLI and helper changes can now begin in parallel.

---

## Phase 3: User Story 1 — Train with Minimal New Config (Priority: P1) 🎯 MVP

**Goal**: A training run launches successfully using only `--dataset` and `--training-info` CLI args, with base model, inference, and publishing configs resolved from reference fields inside the training config.

**Independent Test**: Run `classification-trainer train --dataset <name> --training-info <name>` (where the training config names valid base model and inference configs) and confirm training launches. Confirm running with a non-existent referenced config name prints a clear error and exits non-zero.

### Tests for User Story 1

- [x] T005 [P] [US1] Create `tests/unit/test_training_info.py`: add unit tests for `TrainingInfo` resolution properties — test that `base_model_info`, `inference_info`, and `publishing_info` properties load the correct config objects given valid names; test that `FileNotFoundError` is raised with a descriptive message when a referenced config name does not exist on disk
- [x] T006 [P] [US1] Add tests to `tests/unit/test_training_info.py`: verify that `TrainingInfo` loads correctly from a YAML dict that includes the new fields (`base_model`, `inference`, `publishing`, `model_card_description`) and that `model_card_description` is accessible as a plain string attribute

### Implementation for User Story 1

- [x] T007 [US1] Update `publishing_helper.py`: change the call site at line 192 to read `training_info.model_card_description` instead of `publishing_info.description`; update the affected function signature to pass `training_info` if it is not already a parameter in `src/classification_trainer/helpers/publishing_helper.py`
- [x] T008 [US1] Simplify `train` command in `src/classification_trainer/console/main.py`: remove `--base-model`, `--inference-info`, and `--publishing-info` Typer arguments; load `bm_info`, `inf_info`, and `pub_info` from `tr_info.base_model_info`, `tr_info.inference_info`, and `tr_info.publishing_info` properties instead (depends on T002, T007)
- [x] T009 [P] [US1] Simplify `sweep` command in `src/classification_trainer/console/main.py`: remove `--base-model` and `--inference-info` Typer arguments; load from `tr_info` properties (depends on T002; do after T008 to avoid conflicts on same file)
- [x] T010 [P] [US1] Simplify `analyze-dataset` command in `src/classification_trainer/console/main.py`: remove `--base-model` Typer argument; load from `tr_info.base_model_info` property (depends on T002; do after T009)
- [x] T011 [P] [US1] Simplify `compute-batch-size` command in `src/classification_trainer/console/main.py`: remove `--base-model` Typer argument; load from `tr_info.base_model_info` property (depends on T002; do after T010)
- [x] T012 [US1] Migrate `training_info/reddit-rpg-rules-questions-classifier.yaml`: add `base_model: qwen2.5-0.5b-instruct`, `inference: simple-classification`, `publishing: reddit-rpg-rules-question-classifier`, and `model_card_description` (copy description text from `publishing_info/reddit-rpg-rules-question-classifier.yaml`)
- [x] T013 [P] [US1] Migrate `training_info/imdb.yaml`: add `base_model`, `inference`, `publishing`, and `model_card_description` fields with appropriate values
- [x] T014 [P] [US1] Migrate `publishing_info/reddit-rpg-rules-question-classifier.yaml`: remove the `description` field entirely (depends on T012 being complete so the text is already copied)
- [x] T015 [P] [US1] Rename `dataset_info/rpg_reddit_post_classification.yaml` → `dataset_info/rpg-reddit-post-classification.yaml` (kebab-case per FR-010); verify no references to the old filename remain in any YAML or code files
- [x] T016 [P] [US1] Verify `tests/unit/test_cli.py` still passes after CLI changes — update if any assertion relies on `--base-model` or `--inference-info` appearing in help output

**Checkpoint**: At this point, `classification-trainer train --dataset <name> --training-info <name>` works end-to-end. All three MVP acceptance scenarios are testable.

---

## Phase 4: User Story 2 — Discover and Browse Reusable Configs (Priority: P2)

**Covered by Phase 3 (US1)**: US2 requires that the directory layout clearly separates model-specific configs from reusable ones, and that reusable configs can be selected by name alone. This is fully achieved by:
- The migration tasks (T012–T015) which produce clean, consistently named, reusable config files in their respective directories
- The naming convention (kebab-case) established by T015
- The reference field pattern (T002, T008–T011) which makes the separation explicit in the training config

No additional implementation tasks are needed for US2. Its acceptance scenarios are verified by inspecting the config directories after Phase 3 is complete.

---

## Phase 5: User Story 3 — Author a New Config Using the Example (Priority: P3)

**Goal**: Every config directory contains an `example.yaml` with inline comments for every field, covering: purpose, required vs optional, allowed values, and a concrete example where ambiguous.

**Independent Test**: Open any `example.yaml`, copy it, fill in values guided only by the inline comments, and confirm the resulting file is accepted by the system without modification. No source code reading required.

### Implementation for User Story 3

- [x] T017 [P] [US3] Rewrite `training_info/example.yaml`: add full inline comments for all existing fields AND the four new fields (`base_model`, `inference`, `publishing`, `model_card_description`); each comment must state purpose, required/optional (with default), and allowed values
- [x] T018 [P] [US3] Rewrite `dataset_info/example.yaml`: add full inline comments for every field in `DatasetInfo` (content column, label column, split names, positive case, search settings, etc.)
- [x] T019 [P] [US3] Rewrite `base_model_info/example.yaml`: add full inline comments for `huggingface_name` (format, instruct-only restriction) and `chat_template` (must match a filename stem in `chat_template_info/`)
- [x] T020 [P] [US3] Rewrite `inference_info/example.yaml`: add full inline comments for all fields including `do_sample`, `temperature`, `top_p`, `max_new_tokens`, `repetition_penalty`, `prepare_unsloth_inference`, `metrics` (allowed values list), `sweep_metric`, `sweep_metric_goal`
- [x] T021 [P] [US3] Rewrite `publishing_info/example.yaml`: remove `description` field; add full inline comments for `gguf_quantizations`, `merged_save_method` (allowed values: `merged_16bit`, `merged_4bit`, `lora_merged_16bit`), and all six save/publish flags
- [x] T022 [P] [US3] Rewrite `chat_template_info/example.yaml`: add full inline comments for all fields; explain the relationship between `chat_template` in `base_model_info` and the filename stem in `chat_template_info/`

**Checkpoint**: All six example files are self-contained references. A new user can author any config type without reading source code.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T023 Run `cd src && pytest` and confirm all tests pass with zero failures or errors
- [x] T024 [P] Run `cd src && ruff check .` and confirm no linting errors introduced by the model and helper changes
- [x] T025 [P] Verify `classification-trainer train --help`, `classification-trainer sweep --help`, `classification-trainer analyze-dataset --help`, and `classification-trainer compute-batch-size --help` no longer show `--base-model`, `--inference-info`, or `--publishing-info` arguments
- [x] T026 [P] Update `CLAUDE.md` active technologies and project structure sections to reflect that `training_info` now references `base_model_info`, `inference_info`, and `publishing_info` by name

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — **BLOCKS all user story work**
- **US1 (Phase 3)**: Depends on Foundational (Phase 2) complete
- **US2 (Phase 4)**: No tasks — achieved by US1
- **US3 (Phase 5)**: Independent of all phases — can run in parallel with Phase 3
- **Polish (Phase 6)**: Depends on Phase 3 and Phase 5 complete

### User Story Dependencies

- **US1 (P1)**: Requires Foundational complete; no dependency on other stories
- **US3 (P3)**: Fully independent — example file rewrites touch different files than code changes

### Within Phase 3 (US1)

- T005, T006 (tests): Write before implementation; run to confirm they fail first
- T007 (helper): Requires T002 complete (model_card_description field must exist)
- T008–T011 (CLI): Requires T002 complete; edit `main.py` sequentially (same file — do not parallelize)
- T012–T015 (YAML migration): Requires T002 complete to know final schema; T014 depends on T012 text copy
- T016 (test update): After T008–T011 complete

### Parallel Opportunities

- T003 and T004 (Phase 2) can run in parallel with T002 (different files)
- T005 and T006 (test stubs) can be written while T002 is in progress
- T012–T015 and T017–T022 can all run in parallel (all different files)
- All of Phase 5 (US3) tasks T017–T022 run fully in parallel

---

## Parallel Example: Phase 3 (US1)

```bash
# Write tests first (in parallel):
Task T005: tests/unit/test_training_info.py — resolution property tests
Task T006: tests/unit/test_training_info.py — new field loading tests

# Once T002 (model) is done, run in parallel across different files:
Task T007: helpers/publishing_helper.py — description source change
Task T012: training_info/reddit-rpg-rules-questions-classifier.yaml — add new fields
Task T013: training_info/imdb.yaml — add new fields
Task T015: dataset_info/ — rename rpg file

# CLI changes (same file — sequential):
Task T008 → T009 → T010 → T011: console/main.py — one command at a time
```

## Parallel Example: Phase 5 (US3)

```bash
# All six example files in parallel:
Task T017: training_info/example.yaml
Task T018: dataset_info/example.yaml
Task T019: base_model_info/example.yaml
Task T020: inference_info/example.yaml
Task T021: publishing_info/example.yaml
Task T022: chat_template_info/example.yaml
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002–T004) — **CRITICAL**
3. Complete Phase 3: US1 (T005–T016)
4. **STOP and VALIDATE**: Run `pytest`, confirm `train --help` shows only 2 required args, run one end-to-end train invocation
5. Phase 5 (example files) can follow without risk to running system

### Incremental Delivery

1. Phase 1 + 2 → Models correct, dirs auto-created
2. Phase 3 → Full CLI simplification and YAML migration working (MVP)
3. Phase 5 → All example files documented
4. Phase 6 → Clean bill of health from tests and linter

---

## Notes

- T008–T011 all edit `src/classification_trainer/console/main.py` — do sequentially to avoid conflicts
- `publish` command intentionally retains `--publishing-info` (see contracts/cli.md rationale)
- When migrating imdb configs (T013), check both `training_info/imdb.yaml` and whether a corresponding real publishing config exists; create one if needed or set `publishing: null`
- The `description` field in `PublishingInfo` uses `extra="forbid"` — after T003, any YAML that still has `description` will fail to load; T014 removes it from the real config before that becomes a problem
- Kebab-case is convention only — no Pydantic validator needed (Principle V: Simplicity)
