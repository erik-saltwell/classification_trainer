# Tasks: WandB Hyperparameter Sweep Command

**Input**: Design documents from `/specs/001-wandb-sweep/`
**Prerequisites**: plan.md, spec.md, data-model.md, contracts/sweep-command.md, quickstart.md

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

---

## Phase 1: Setup (Configuration & Infrastructure)

**Purpose**: Model and path changes that all subsequent phases depend on. No logic yet — pure data/structure.

- [x] T001 Add `sweep_trial_outputs(model_name: str, run_id: str) -> Path` method to `CommonPaths` in `src/classification_trainer/utils/common_paths.py` — returns `OUTPUTS_DIR / model_name / run_id`, NOT added to `ensure_all_dirs_exist()`
- [x] T002 [P] Add `sweep_metric: str = "f1"` and `sweep_metric_goal: str = "maximize"` optional fields to `InferenceInfo` in `src/classification_trainer/configuration/inference_info.py`
- [x] T003 [P] Add optional `output_dir: str | None = None` parameter to `TrainingInfo.create_sft_config()` in `src/classification_trainer/configuration/training_info.py` — when non-None, overrides `self.model_name` as the SFTConfig `output_dir`; all existing callers unaffected

**Checkpoint**: Config models updated — no behaviour change to existing commands yet.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Sweep helper functions and command scaffolding. MUST be complete before any user story work.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T004 Create `src/classification_trainer/helpers/sweep_helper.py` with three pure functions: `build_sweep_config(inference_info: InferenceInfo) -> dict` (calls `SFTParameters.get_default_sweep_config(inference_info.sweep_metric, inference_info.sweep_metric_goal)`); `apply_trial_sft_parameters(training_info: TrainingInfo, trial_config: Mapping[str, Any]) -> TrainingInfo` (uses `SFTParameters.from_dict` + `training_info.model_copy(update=...)`); `extract_target_metric(results: list[MetricResult], metric_name: str) -> float` (finds matching MetricResult, raises ValueError if not found)
- [x] T005 Create `src/classification_trainer/commands/sweep.py` with `SweepCommand` dataclass implementing `CommmandProtocol` — fields: `dataset_info`, `base_model_info`, `training_info`, `inference_info`, `count: int = 10`; stub `execute(logger: LoggingProtocol) -> None` with a `pass` body; imports from sweep_helper and all relevant helpers/configuration modules
- [x] T006 Register `sweep` CLI command in `src/classification_trainer/console/main.py` — add `@app.command("sweep")` function with `--dataset`, `--base-model`, `--training-info`, `--inference-info`, and `--count` arguments, loading configs via `load_config_or_exit` and constructing `SweepCommand(...).execute(logger)`

**Checkpoint**: `classification-trainer sweep --help` works and lists all arguments. No trial logic yet.

---

## Phase 3: User Story 1 — Run a Hyperparameter Sweep (Priority: P1) 🎯 MVP

**Goal**: A user can launch the sweep command and have multiple training trials run automatically, each with different SFT hyperparameters, with results appearing ranked in the wandb dashboard.

**Independent Test**: Run `classification-trainer sweep --dataset imdb --base-model qwen2.5-0.5b-instruct --training-info imdb --inference-info simple_classification --count 2` and confirm: 2 trials complete; each trial has different `sft_parameters` values visible in wandb; both appear in the wandb sweep dashboard; the trial's classification metrics (accuracy, precision, recall, f1) are logged.

- [x] T007 [US1] Implement `_run_trial()` inner function inside `SweepCommand.execute()` in `src/classification_trainer/commands/sweep.py` — the function must: call `wandb.init(project=..., config=sft_params_only_dict)`; read `wandb.config`; call `apply_trial_sft_parameters(self.training_info, dict(wandb.config))` to get a trial-specific `TrainingInfo`; load tokenizer via `load_tokenizer_from_hf`; load dataset via `load_dataset_from_hf`, prep splits; call `load_base_model` with the trial TrainingInfo; call `create_trainer` / `run_training`; run classifier inference evaluation identical to `TrainCommand.test_model` (setup_unsloth_inference → add_inferred_column → add_classification_result_column → collect_classification_counts → generate_metrics); log all metrics via `WandBMetricsReporter` at `step = final_global_step + 1`; call `wandb.finish()`
- [x] T008 [US1] Implement per-trial output directory isolation in `_run_trial()` in `src/classification_trainer/commands/sweep.py` — after `wandb.init()`, pass `output_dir=str(CommonPaths.get().sweep_trial_outputs(self.training_info.model_name, wandb.run.id))` to `trial_training_info.create_sft_config()` via `create_trainer`; requires T001 and T003
- [x] T009 [US1] Implement `SweepCommand.execute()` outer body in `src/classification_trainer/commands/sweep.py` — validate `self.training_info.wandb_config is not None` (exit with error if absent); call `build_sweep_config(self.inference_info)` from sweep_helper; call `wandb.sweep(sweep_config, project=wandb_config.project, entity=None)`; log the sweep ID to the logger; call `wandb.agent(sweep_id, _run_trial, count=self.count)`

**Checkpoint**: US1 fully testable — sweep runs, trials complete, results appear in wandb ranked by the target metric.

---

## Phase 4: User Story 2 — Configure the Target Quality Metric (Priority: P2)

**Goal**: The sweep optimises for whichever metric is set in the inference config YAML. Invalid metric names are caught before any trials run.

**Independent Test**: (a) Set `sweep_metric: "accuracy"` in the inference YAML, run sweep, confirm wandb sweep page shows "accuracy" as the optimisation objective. (b) Set `sweep_metric: "not_a_metric"`, run sweep, confirm the command exits immediately with a descriptive error and no trials start.

- [x] T010 [US2] Add startup validation in `SweepCommand.execute()` in `src/classification_trainer/commands/sweep.py` — before calling `wandb.sweep()`, validate: (1) `self.inference_info.sweep_metric` is in `_METRIC_REGISTRY` (import from evaluation_helper); if not, print error listing valid options and raise `typer.Exit(code=1)`; (2) `self.inference_info.sweep_metric_goal` is `"maximize"` or `"minimize"`; if not, print error and raise `typer.Exit(code=1)`; (3) if `sweep_metric` was not set in YAML (defaulted to `"f1"`), log a notice to the user identifying the default being used
- [x] T011 [P] [US2] Update `inference_info/simple_classification.yaml` to add `sweep_metric: "f1"` and `sweep_metric_goal: "maximize"` with a comment stating valid metric names

**Checkpoint**: US2 fully testable — metric is configurable from YAML; invalid names are caught at startup.

---

## Phase 5: User Story 3 — Apply Best Configuration to a Training Run (Priority: P3)

**Goal**: Hyperparameter names shown in the wandb sweep results match exactly the field names in the `sft_parameters` block of the training YAML, so the user can copy values directly.

**Independent Test**: Complete a sweep; open the best trial in wandb; confirm every parameter key displayed (e.g., `rank`, `learning_rate`, `lr_scheduler_type`) exactly matches a field name in `SFTParameters`; create a training YAML using those values; run `classification-trainer train` with that YAML and confirm it completes without validation errors.

- [x] T012 [US3] Fix typo in `SFTParameters.get_default_sweep_config()` in `src/classification_trainer/configuration/sft_parameters.py` — rename key `"lr_schedular_type"` → `"lr_scheduler_type"` to match the actual `SFTParameters.lr_scheduler_type` field name (currently broken: wandb would show the misspelled key which cannot be copied into a valid training YAML)
- [x] T013 [US3] Audit all keys in `SFTParameters.get_default_sweep_config()["parameters"]` in `src/classification_trainer/configuration/sft_parameters.py` — verify each key matches a field name in `SFTParameters`; fix any remaining mismatches; confirm the complete set of keys is: `rank`, `alpha_multiplier`, `use_projection_modules`, `warmup_ratio`, `lr_scheduler_type`, `optim`, `learning_rate`, `lora_dropout`, `weight_decay`

**Checkpoint**: US3 fully testable — all sweep parameter keys map directly to training YAML fields.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Cleanup and final verification across all stories.

- [x] T014 [P] Verify `src/classification_trainer/commands/__init__.py` exports `SweepCommand` if other commands are exported there; add export if needed
- [x] T015 Run `ruff check .` from `src/` and fix any linting errors introduced by new files (`sweep_helper.py`, `sweep.py`) and modified files

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — T002 and T003 can run in parallel; T001 is independent
- **Foundational (Phase 2)**: T004 → T005 → T006 (each depends on previous)
- **User Story phases (3–5)**: All depend on Phase 2 completion
- **Polish (Phase 6)**: Depends on all story phases being complete

### User Story Dependencies

- **US1 (Phase 3)**: Depends on Phase 1 (T001, T003) and Phase 2; no dependency on US2 or US3
- **US2 (Phase 4)**: Depends on Phase 1 (T002) and Phase 2; T011 is independent of T010
- **US3 (Phase 5)**: Depends on Phase 1 only (sft_parameters.py); can be worked in parallel with US1 and US2

### Within Each User Story

- **US1**: T007 → T008 → T009 (sequential — all in sweep.py, each builds on previous)
- **US2**: T010 then T011 [P] (T011 is a YAML file, independent of T010)
- **US3**: T012 → T013 (T013 audits after T012 fixes the known typo)

### Parallel Opportunities

- **Phase 1**: T001, T002, T003 can all run in parallel (different files)
- **US2**: T011 (YAML update) can run in parallel with T010 (sweep.py validation)
- **US3**: Both tasks can begin as soon as Phase 1 is done, independent of US1 and US2

---

## Parallel Execution Examples

### Phase 1 (all three can start simultaneously)

```
T001: src/classification_trainer/utils/common_paths.py
T002: src/classification_trainer/configuration/inference_info.py
T003: src/classification_trainer/configuration/training_info.py
```

### After Phase 2 completes (US3 can start immediately, independent of US1/US2)

```
T007–T009: src/classification_trainer/commands/sweep.py  (US1)
T010–T011: src/classification_trainer/commands/sweep.py + inference_info YAML  (US2)
T012–T013: src/classification_trainer/configuration/sft_parameters.py  (US3)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T006)
3. Complete Phase 3: US1 (T007–T009)
4. **STOP and VALIDATE**: Run a 2-trial sweep against imdb dataset; confirm trials appear in wandb ranked by target metric
5. US2 and US3 are quality improvements — the sweep is functional without them

### Incremental Delivery

1. Phase 1 + Phase 2 → `sweep --help` works
2. Phase 3 → Sweep runs trials and reports to wandb (MVP)
3. Phase 4 → Metric is configurable and validated
4. Phase 5 → Best config is copy-pasteable into training YAML
5. Phase 6 → Linting and export cleanup

### Notes

- The `suppress_wandb_finish` mechanism in `run_training` is activated automatically when `wandb.run is not None` — the trial function must call `wandb.init()` before `run_training` to ensure this suppression is active
- Metric logging step: use `final_global_step + 1` (same pattern as `TrainCommand`); `WandBMetricsReporter.report(results, step=final_step+1)` logs all metrics including the sweep target metric in one call — no separate `wandb.log()` needed
- `SFTParameters` fields used by `get_default_sweep_config()` keys map to the `sft_parameters:` block in training YAML — the user copies values at that nesting level, not at the top level of the YAML
