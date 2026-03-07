# Tasks: User-Configurable Sweep Parameters

**Input**: Design documents from `/specs/004-sweep-config/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Tests are included — the spec requires validation of edge cases and the data model has complex validation rules that warrant unit tests.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Create the new configuration model file and test file

- [x] T001 [P] Create `SweepParameterSpec` Pydantic model in `src/classification_trainer/configuration/sweep_config.py` with three variant fields (`values: list | None`, `min: float | None`, `max: float | None`, `value: scalar | None`), a `model_validator` that enforces exactly one format is provided, and validators for: non-empty values list, min < max, min/max co-presence. Include a `SweepMethod` StrEnum with values `random`, `bayes`, `grid`.
- [x] T002 [P] Create empty test file `tests/unit/test_sweep_config.py` with imports for `SweepParameterSpec`, `SweepConfig`, and `pytest`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Complete the `SweepConfig` model and wire it into `TrainingInfo` — MUST be done before any user story work

- [x] T003 Create `SweepConfig` Pydantic model in `src/classification_trainer/configuration/sweep_config.py` with fields `method: SweepMethod = SweepMethod.RANDOM` and `parameters: dict[str, SweepParameterSpec]`. Add a `model_validator` that: (a) rejects empty `parameters` dict, (b) validates all keys against `SFTParameters.model_fields.keys()`, (c) rejects `grid` method when any parameter uses min/max format. Export both models from `src/classification_trainer/configuration/__init__.py`.
- [x] T004 Add `sweep: SweepConfig | None = None` field to `TrainingInfo` in `src/classification_trainer/configuration/training_info.py`. Import `SweepConfig` from the new module. Ensure `sweep` defaults to `None` so existing configs without a sweep block continue to load unchanged.
- [x] T005 Write unit tests in `tests/unit/test_sweep_config.py` covering: valid discrete list, valid min/max range, explicit value field (`SweepParameterSpec(value=32)`) accepted, empty values list rejected, min >= max rejected, min without max rejected, invalid parameter name rejected, empty parameters dict rejected, grid + continuous range rejected. Use Pydantic's `ValidationError` for rejection assertions.
- [x] T006 Add domain validation logic in `SweepConfig`'s `model_validator` in `src/classification_trainer/configuration/sweep_config.py`: for each parameter in `parameters`, validate that discrete `values` and fixed `value` entries are acceptable for that `SFTParameters` field — specifically, `optim` values must be valid `OptimizerType` strings, `lr_scheduler_type` values must be valid `LRSchedulerType` strings, `lora_dropout` and `warmup_ratio` values must be between 0.0 and 1.0, `rank` and `alpha_multiplier` must be positive integers. For min/max ranges, validate that min and max fall within the field's valid domain. Add corresponding test cases in `tests/unit/test_sweep_config.py`: `lora_dropout: {values: [1.5]}` rejected, `optim: {values: ["not_real"]}` rejected, `rank: {values: [-1]}` rejected, valid values accepted.

**Checkpoint**: `SweepConfig` and `SweepParameterSpec` models are complete with full validation including domain constraints. `TrainingInfo` accepts optional `sweep` block. All validation edge cases have passing tests.

---

## Phase 3: User Story 1 - Customize Sweep Hyperparameter Ranges (Priority: P1)

**Goal**: Users can add a `sweep` block to their training config YAML to control which parameters are swept and with what ranges. Unlisted parameters use `sft_parameters` fixed values (opt-in semantics).

**Independent Test**: Add a `sweep` block with custom `rank` and `learning_rate` ranges to a training config, run a sweep with 3 trials, confirm wandb trials use the specified ranges and hold other params constant.

### Implementation for User Story 1

- [x] T007 [US1] Add `to_wandb_param_spec(param_name: str) -> dict` method to `SweepParameterSpec` in `src/classification_trainer/configuration/sweep_config.py`. This method translates the user's YAML format to wandb's parameter dict: `values` → `{"values": [...]}`, `min/max` → `{"distribution": "log_uniform_values"|"uniform", "min": X, "max": Y}` (log-uniform for `learning_rate`, uniform for others), `value` → `{"value": X}`.
- [x] T008 [US1] Add `to_wandb_sweep_config(sft_parameters: SFTParameters, metric_name: str, metric_goal: str) -> dict` method to `SweepConfig` in `src/classification_trainer/configuration/sweep_config.py`. This method builds the full wandb sweep config dict: sets `method`, `metric`, and `parameters`. For each `SFTParameters` field: if the field is in `self.parameters`, use its `to_wandb_param_spec()`; otherwise, emit `{"value": <sft_parameters value>}` as a fixed constant.
- [x] T009 [US1] Update `build_sweep_config()` in `src/classification_trainer/helpers/sweep_helper.py` to accept `training_info: TrainingInfo` instead of just `inference_info`. When `training_info.sweep` is not `None`, call `training_info.sweep.to_wandb_sweep_config(training_info.sft_parameters, ...)`. When `sweep` is `None`, fall back to `SFTParameters.get_default_sweep_config()` for backward compatibility.
- [x] T010 [US1] Update the call to `build_sweep_config()` in `src/classification_trainer/commands/sweep.py` to pass `training_info` (and `inference_info` for metric name/goal). Ensure backward compatibility path works when no `sweep` block is present.
- [x] T011 [US1] Write unit tests in `tests/unit/test_sweep_config.py` for `to_wandb_param_spec()`: discrete list produces `{"values": [...]}`, min/max with `learning_rate` produces `{"distribution": "log_uniform_values", ...}`, min/max with other params produces `{"distribution": "uniform", ...}`, fixed scalar produces `{"value": X}`.
- [x] T012 [US1] Write unit tests in `tests/unit/test_sweep_config.py` for `to_wandb_sweep_config()`: listed params appear with their sweep spec, unlisted params appear as `{"value": <sft_parameters_default>}`, method and metric fields are correct.

**Checkpoint**: Sweep command uses user-defined parameter ranges when `sweep` block is present; falls back to hardcoded defaults when absent. All SFT parameter fields appear in the wandb config (swept or fixed).

---

## Phase 4: User Story 2 - Fix a Parameter to a Single Value (Priority: P2)

**Goal**: Users can set a bare scalar value for any parameter in the sweep block to hold it constant across all trials.

**Independent Test**: Set `optim: "adamw_bnb_8bit"` as a bare scalar in the sweep block, run 5 trials, confirm every trial uses that exact optimizer.

### Implementation for User Story 2

- [x] T013 [US2] Add a Pydantic `model_validator` (or adjust the YAML parsing in `SweepConfig`) in `src/classification_trainer/configuration/sweep_config.py` that detects bare scalar values in the raw `parameters` dict and wraps them as `SweepParameterSpec(value=X)` before validation. This enables YAML like `optim: "adamw_bnb_8bit"` alongside the dict formats `rank: {values: [8, 16]}`.
- [x] T014 [US2] Write unit tests in `tests/unit/test_sweep_config.py` for bare scalar parsing: integer scalar (`rank: 32`), string scalar (`optim: "adamw_bnb_8bit"`), float scalar (`learning_rate: 0.0002`), boolean scalar (`use_projection_modules: false`). Verify each produces a `SweepParameterSpec` with `value` set and all other fields `None`.

**Checkpoint**: Bare scalar syntax works in YAML for fixing parameters. The `to_wandb_param_spec()` method (from US1) already handles the `value` field, so no additional wandb translation work is needed.

---

## Phase 5: User Story 3 - Select a Sweep Search Method (Priority: P3)

**Goal**: Users can set `method: random`, `method: bayes`, or `method: grid` in the sweep block to control the search strategy.

**Independent Test**: Run two sweeps with different method settings, confirm wandb dashboard shows the correct method for each.

### Implementation for User Story 3

- [x] T015 [US3] Write unit tests in `tests/unit/test_sweep_config.py` for method selection: `to_wandb_sweep_config()` with `method=bayes` produces `{"method": "bayes", ...}`, `method=grid` with all-discrete params produces `{"method": "grid", ...}`, default method is `random`, `grid` with a min/max param raises `ValidationError`.

**Checkpoint**: Method selection is already implemented via the `SweepMethod` enum (Phase 2) and `to_wandb_sweep_config()` (US1). This phase adds targeted test coverage to confirm correct behavior. The grid+continuous validation was added in Phase 2.

---

## Phase 6: User Story 4 - Use Continuous Distributions for Numeric Parameters (Priority: P4)

**Goal**: Users can specify `{min: X, max: Y}` for numeric parameters. The system auto-selects log-uniform for `learning_rate` and uniform for all others.

**Independent Test**: Specify `learning_rate: {min: 1e-5, max: 1e-3}` and `weight_decay: {min: 0.0, max: 0.1}`, run a sweep, confirm wandb samples from continuous ranges.

### Implementation for User Story 4

- [x] T016 [US4] Write unit tests in `tests/unit/test_sweep_config.py` for distribution auto-selection: `to_wandb_param_spec("learning_rate")` with min/max produces `log_uniform_values`, `to_wandb_param_spec("weight_decay")` with min/max produces `uniform`, `to_wandb_param_spec("lora_dropout")` with min/max produces `uniform`, `to_wandb_param_spec("warmup_ratio")` with min/max produces `uniform`.

**Checkpoint**: Distribution auto-selection is already implemented in `to_wandb_param_spec()` (US1). This phase adds targeted test coverage for each numeric parameter type.

---

## Phase 7: User Story 5 - Monitor Sweep Progress and Verify Trial Parameters (Priority: P5)

**Goal**: Each trial prints its number ("Trial 3 of 10") and all training parameter values before training begins.

**Independent Test**: Run a sweep with 3 trials, confirm terminal output shows trial counter and parameter values for each trial.

### Implementation for User Story 5

- [x] T017 [US5] Add a `_trial_number: int = 0` instance variable to `SweepCommand` in `src/classification_trainer/commands/sweep.py`. At the start of `_run_trial()`, increment the counter and log `"Trial {n} of {count}"` via `logger.report_message()`.
- [x] T018 [US5] After `apply_trial_sft_parameters()` in `_run_trial()` in `src/classification_trainer/commands/sweep.py`, log all trial parameter names and values by calling `trial_training_info.sft_parameters.to_dict()` and formatting each key-value pair. Use `logger.report_message()` with a "Trial parameters:" header followed by indented key-value lines.

**Checkpoint**: Terminal output shows trial progress counter and parameter values for each trial before training begins.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and final validation

- [x] T019 [P] Add a fully commented `sweep` block to `training_info/example.yaml` documenting: the `method` field with all three allowed values, the `parameters` sub-block with examples of all three parameter formats (discrete list, continuous range, fixed scalar), and inline comments explaining opt-in semantics (unlisted params use `sft_parameters` values).
- [x] T020 Run all tests with `cd src && pytest` to verify no regressions across all existing test files and new `test_sweep_config.py`.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user stories
- **User Stories (Phase 3–7)**: All depend on Phase 2 completion
  - US1 (Phase 3): Core implementation — recommended first
  - US2 (Phase 4): Depends on US1 (bare scalar needs `to_wandb_param_spec`)
  - US3 (Phase 5): Independent of US2, depends on US1
  - US4 (Phase 6): Independent of US2/US3, depends on US1
  - US5 (Phase 7): Independent of US2/US3/US4, depends on US1 (needs `_run_trial` changes)
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: Foundational only — no other story dependencies. **MVP.**
- **US2 (P2)**: Depends on US1 (`to_wandb_param_spec` must handle `value` field)
- **US3 (P3)**: Depends on US1 (`to_wandb_sweep_config` must emit `method`). Can parallelize with US2.
- **US4 (P4)**: Depends on US1 (distribution auto-selection in `to_wandb_param_spec`). Can parallelize with US2/US3.
- **US5 (P5)**: Depends on US1 (trial function structure). Can parallelize with US2/US3/US4.

### Parallel Opportunities

- T001 and T002 can run in parallel (different new files)
- US3, US4, US5 can run in parallel after US1 (each touches different concerns)
- T019 can run in parallel with any user story phase (documentation only)

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001, T002)
2. Complete Phase 2: Foundational (T003, T004, T005, T006)
3. Complete Phase 3: User Story 1 (T007–T012)
4. **STOP and VALIDATE**: Test with a training config containing a `sweep` block and one without
5. Verify wandb sweep uses custom ranges / falls back to defaults

### Incremental Delivery

1. Setup + Foundational → Models and validation ready
2. US1 → Custom parameter ranges work → **MVP**
3. US2 → Bare scalar syntax for fixing params
4. US3 → Method selection (random/bayes/grid)
5. US4 → Continuous distribution auto-selection confirmed
6. US5 → Trial progress and parameter display
7. Polish → Documentation in example.yaml, full test run

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US2–US4 are lightweight — mostly test tasks confirming behavior already implemented in US1
- US5 is the only story that modifies `commands/sweep.py` directly
- The existing `get_default_sweep_config()` is preserved for backward compatibility — do not remove it
- Commit after each phase completion
