# Tasks: Sweep Parameter Simplification and Observability

**Input**: Design documents from `/specs/010-sweep-params-observability/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅

**Tests**: Test updates are included because existing tests break and new behaviour must be verified.

**Organization**: Tasks grouped by user story. US1 has foundational prerequisites; US2 and US3 are additive logging changes on top of US1.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no shared dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

---

## Phase 1: Setup

No project setup required. No new files, directories, or dependencies are introduced. Proceed directly to Phase 2.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core changes that must land before US1 tests can be written and before any call site can be updated. All three tasks touch different files and can be done in parallel.

**⚠️ CRITICAL**: US1 implementation tasks depend on T001 and T002 being complete.

- [x] T001 [P] Simplify `to_wandb_sweep_config` in `src/classification_trainer/configuration/sweep_info.py`: remove the `sft_parameters: SFTParameters` parameter entirely; replace the full-field loop with a dict comprehension over `self.parameters` only; remove the now-unused `SFTParameters` import (keep `LRSchedulerType` and `OptimizerType`)
- [x] T002 [P] Fix `apply_trial_sft_parameters` in `src/classification_trainer/helpers/sweep_helper.py`: merge `dict(trial_config)` onto `training_info.sft_parameters.to_dict()` (wandb values win), then construct `SFTParameters.from_dict(merged)` — replaces the current `SFTParameters.from_dict(dict(trial_config))` which fails when only swept fields are present
- [x] T003 [P] Update sweep_config comment block in `training_info/example.yaml`: replace the "Fixed value" format description with a note that unlisted parameters automatically use their `sft_parameters` values; remove the `optim: "adamw_bnb_8bit"` fixed-value example from the `parameters` example

**Checkpoint**: These three files are updated. No call sites or tests touch the new signatures yet — existing tests will be failing after T001. Proceed to Phase 3.

---

## Phase 3: User Story 1 — Swept Parameters Only (Priority: P1) 🎯 MVP

**Goal**: The sweep config `parameters` block only lists fields being varied. wandb receives only those fields. Trial parameters merge wandb values onto base `sft_parameters`.

**Independent Test**: Configure a sweep with only `rank` and `learning_rate` in `parameters`. Verify `to_wandb_sweep_config()` output contains exactly those two fields. Verify `apply_trial_sft_parameters` with `{"rank": 32}` produces `sft_parameters.rank == 32` and all other fields unchanged from base.

### Implementation for User Story 1

- [x] T004 [US1] Update `generate_sweep_parameters` in `src/classification_trainer/commands/sweep.py`: change `self.training_info.sweep_config.to_wandb_sweep_config(self.training_info.sft_parameters)` to `self.training_info.sweep_config.to_wandb_sweep_config()` (drop the positional `sft_parameters` argument)
- [x] T005 [US1] Update `tests/unit/test_sweep_info.py` — call sites: remove the `sft_parameters` positional argument from every `to_wandb_sweep_config(sft, ...)` call; update any assertions that check the output `parameters` dict keys to expect only the swept fields (not all SFT fields)
- [x] T006 [US1] Add `apply_trial_sft_parameters` merge tests to `tests/unit/test_sweep_info.py` (after T005, same file): add three test functions — (a) single swept field overrides base while others are preserved, (b) multiple swept fields all override correctly, (c) empty `trial_config` leaves base unchanged

**Checkpoint**: `uv run pytest tests/` passes. US1 is fully functional — sweep YAML only needs swept fields, trial parameters merge correctly.

---

## Phase 4: User Story 2 — Sweep Initialization Logging (Priority: P2)

**Goal**: When a sweep is initialized, the full sweep config submitted to wandb is logged to the console before the first trial begins.

**Independent Test**: Run sweep command with a valid config; confirm a log message containing the sweep config (method, metric, parameter names) appears before any trial output.

### Implementation for User Story 2

- [x] T007 [US2] Add sweep config logging to `src/classification_trainer/commands/sweep.py`: add `import json` at top of file; in `execute()`, after `parameters = self.generate_sweep_parameters()`, add two `self.logger.report_message()` calls — one for a `[blue]Sweep config submitted to wandb:[/blue]` header, one for `json.dumps(parameters, indent=2)`

**Checkpoint**: Sweep initialization produces a readable JSON log of the sweep definition before trial 1. US2 independently testable by inspecting logger output.

---

## Phase 5: User Story 3 — Per-Trial Parameter Verification Table (Priority: P2)

**Goal**: At the start of each sweep trial, a three-column table is logged showing the parameter name, the wandb-provided value, and the actual trainer value — one row per swept parameter.

**Independent Test**: Run a single sweep trial; confirm the logger output contains a table with header `["Parameter", "WandB Value", "Trainer Value"]` and one row per field in `run.config`, with trainer values matching the merged `sft_parameters`.

### Implementation for User Story 3

- [x] T008 [US3] Add per-trial parameter table to `run_single_trial` in `src/classification_trainer/commands/sweep.py`: after `self.runner.training_info = apply_trial_sft_parameters(...)`, build `trial_sft = self.runner.training_info.sft_parameters.to_dict()`; construct `rows` as `[[name, str(wandb_val), str(trial_sft.get(name, "N/A"))] for name, wandb_val in sweep_run_config.items()]`; call `self.logger.report_message("[blue]Trial sweep parameters:[/blue]")` then `self.logger.report_multicolumn_table(headers=["Parameter", "WandB Value", "Trainer Value"], rows=rows)`

**Checkpoint**: Each trial produces a parameter comparison table in the log. US3 independently testable by inspecting per-trial logger output.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T009 Run `uv run pytest tests/` and confirm all tests pass (0 failures)
- [x] T010 [P] Run `uv run ruff check .` and fix any issues introduced by the changes
- [x] T011 [P] Run `uv run pyright src/` and confirm 0 errors (verify no stray `sft_parameters` arg references remain anywhere)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Foundational (Phase 2)**: No dependencies — start immediately. T001, T002, T003 are all parallel.
- **US1 (Phase 3)**: Depends on T001 (T004, T005) and T002 (T006). T005 must precede T006 (same file).
- **US2 (Phase 4)**: Depends on T004 (same file, `sweep.py`). Can start after T004.
- **US3 (Phase 5)**: Depends on T007 (same file, sequential edits to `sweep.py`).
- **Polish (Phase 6)**: Depends on all story phases complete. T010 and T011 are parallel.

### User Story Dependencies

- **US1 (P1)**: Depends on T001 and T002. Independent of US2/US3.
- **US2 (P2)**: Depends on T004 (US1). Logically correct only after US1 (logged config reflects simplified params).
- **US3 (P2)**: Depends on T007 (US2). Same file, sequential.

### Within Each User Story

- US1: T004 → T005 → T006 (T004 and T005 depend on T001; T006 depends on T002 and must follow T005 in same file)
- US2: T007 (single task, depends on T004)
- US3: T008 (single task, depends on T007)

### Parallel Opportunities

```bash
# Phase 2 — all three in parallel:
Task T001: "Simplify to_wandb_sweep_config in sweep_info.py"
Task T002: "Fix apply_trial_sft_parameters in sweep_helper.py"
Task T003: "Update example.yaml sweep_config comment"

# Phase 6 — linters in parallel after pytest:
Task T010: "ruff check ."
Task T011: "pyright src/"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 2: Foundational (T001, T002, T003 in parallel)
2. Complete Phase 3: US1 (T004 → T005 → T006)
3. **STOP and VALIDATE**: `uv run pytest tests/` passes; sweep YAML with 2 params works end-to-end
4. Proceed to US2/US3 only after US1 is confirmed

### Incremental Delivery

1. T001+T002+T003 → Foundation ready (3 parallel tasks)
2. T004+T005+T006 → US1 complete: simplified YAML, correct trial merging
3. T007 → US2 complete: sweep init visibility
4. T008 → US3 complete: per-trial visibility
5. T009+T010+T011 → All clean

### Notes

- T001 will cause existing `test_sweep_info.py` tests to fail immediately — this is expected. T005 fixes them.
- T002 is safe to merge before T006 (no existing tests cover `apply_trial_sft_parameters` with partial config).
- `sweep.py` is touched by T004, T007, and T008 — keep edits sequential to avoid conflicts.
- All `[P]` tasks touch different files and have no shared incomplete dependencies.
