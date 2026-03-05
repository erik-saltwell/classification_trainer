# Implementation Plan: WandB Hyperparameter Sweep Command

**Branch**: `001-wandb-sweep` | **Date**: 2026-03-05 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-wandb-sweep/spec.md`

## Summary

Add a `sweep` CLI command that uses wandb's sweep agent to run multiple training trials, each with a different combination of `SFTParameters` hyperparameters drawn from `SFTParameters.get_default_sweep_config()`. After each trial's training pass, the command runs a full classifier inference evaluation (same pipeline as `TrainCommand.test_model`) and logs the target classification metric (e.g., F1) to wandb for sweep ranking. The target metric is configured via two new fields on `InferenceInfo`: `sweep_metric` and `sweep_metric_goal`.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: wandb (sweep + agent API), unsloth `FastLanguageModel`, TRL `SFTTrainer`, Pydantic v2, Typer, HuggingFace `datasets`
**Storage**: YAML config files (`inference_info/`, `training_info/`); per-trial checkpoint directories under `outputs/<model_name>/<run_id>/`
**Testing**: pytest
**Target Platform**: Linux with GPU (same as existing commands)
**Project Type**: CLI
**Performance Goals**: No additional performance constraints beyond existing training pipeline
**Constraints**: Trial isolation — model and tokenizer must be loaded fresh per trial; wandb step values must be monotonically increasing within each trial run
**Scale/Scope**: Single-machine sweep agent; number of trials controlled by `--count` argument

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-First | PASS | `InferenceInfo` gains `sweep_metric` + `sweep_metric_goal` fields (YAML-driven). No hardcoded metric names in command or helper. `CommonPaths` needs a `sweep_trial_outputs` property for per-trial directories. |
| II. Protocol-Based Interfaces | PASS | `SweepCommand` implements `CommmandProtocol`. Helpers accept `LoggingProtocol` and `MetricsReportingProtocol` only. |
| III. Separation of Concerns | PASS | `SweepCommand` (commands/): orchestration only — builds sweep config, launches agent, coordinates per-trial closure. `sweep_helper.py` (helpers/): domain logic — builds wandb sweep config dict, applies sweep trial config to TrainingInfo, extracts target metric value from results. All training/eval/inference calls go through existing helpers. |
| IV. Observability | PASS | Each trial logs all classification metrics via `WandBMetricsReporter` at `final_step + 1`. Target metric string matches the sweep config's `metric.name` exactly, so wandb sweep controller can rank trials. Each trial uses the seed from `TrainingInfo`. |
| V. Simplicity & Scope | PASS | No new abstractions. Reuses `load_base_model`, `create_trainer`, `run_training`, `test_model` logic, `setup_unsloth_inference`, `add_inferred_column`, evaluation pipeline. `sweep_helper.py` contains three focused pure functions. |

## Project Structure

### Documentation (this feature)

```text
specs/001-wandb-sweep/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
src/classification_trainer/
├── commands/
│   └── sweep.py                  # NEW: SweepCommand
├── helpers/
│   └── sweep_helper.py           # NEW: build_sweep_config, apply_trial_config, extract_target_metric
├── configuration/
│   └── inference_info.py         # MODIFIED: add sweep_metric, sweep_metric_goal fields
├── console/
│   └── main.py                   # MODIFIED: register sweep CLI command
└── utils/
    └── common_paths.py           # MODIFIED: add sweep_trial_outputs property

inference_info/
└── simple_classification.yaml   # MODIFIED: document new optional fields in example
```

**Structure Decision**: Single-project layout (existing structure). No new top-level directories. `sweep_helper.py` is a peer to `training_helper.py`, `evaluation_helper.py`, etc. Per-trial checkpoint directories live under existing `outputs/` tree, routed through `CommonPaths`.

## Complexity Tracking

No constitution violations.

---

## Phase 0: Research

### R-001: wandb Agent Exception Handling for Trial Isolation (FR-009)

**Decision**: Wrap the trial function body in a `try/except Exception` block. On exception, call `wandb.finish(exit_code=1)` before re-raising or swallowing. The wandb agent treats a non-zero exit code as a failed run and continues to the next trial.

**Rationale**: `wandb.agent()` by default stops the sweep if the trial function raises an unhandled exception. Catching and calling `wandb.finish(exit_code=1)` inside the trial function marks the run as failed while allowing the agent loop to continue. This satisfies FR-009.

**Alternatives considered**: `wandb.agent(..., catch_exceptions=True)` exists but is not available in all wandb versions and swallows errors silently. Explicit try/except is more transparent and compatible.

---

### R-002: Step Counter in Sweep Trial Context (Constitution IV / known bug pattern)

**Decision**: Each trial opens a fresh `wandb.init()` run. The HF trainer starts that run's step counter at 0. Post-training classification metrics MUST be logged at `final_global_step + 1`, matching the existing pattern in `TrainCommand`. The sweep target metric is included in this same `wandb.log()` call via `WandBMetricsReporter`, so no extra log call is needed.

**Rationale**: wandb requires monotonically increasing steps within a single run. Fresh `wandb.init()` per trial means the step counter resets to 0 for each trial — no cross-trial contamination. The `suppress_wandb_finish` mechanism in `run_training` correctly suppresses the trainer's early `wandb.finish()` call because `wandb.run is not None` when the trial function initialises wandb first.

**Alternatives considered**: Logging the sweep target metric separately after `WandBMetricsReporter.report()`. Unnecessary — `WandBMetricsReporter` already calls `wandb.log({metric_name: value}, step=step)`, and the sweep controller uses the last logged value for the configured metric name.

---

### R-003: Sweep Config Metric Name Matching

**Decision**: The wandb sweep config's `metric.name` MUST be set to the exact string stored in `inference_info.sweep_metric` (e.g., `"f1"`). `WandBMetricsReporter` logs metrics using `MetricResult.metric_name` as the dict key. The existing metric classes (`F1Metric`, `AccuracyMetric`, etc.) produce `MetricResult` objects whose `metric_name` field matches the keys in `_METRIC_REGISTRY` (`"f1"`, `"accuracy"`, `"precision"`, `"recall"`, `"total_seen"`). These are the only valid values for `sweep_metric`.

**Rationale**: If the metric name in the sweep config doesn't match what's logged, the wandb controller cannot rank trials. Constraining `sweep_metric` to `_METRIC_REGISTRY` keys at startup (FR-005) ensures alignment.

**Alternatives considered**: Allowing arbitrary strings for `sweep_metric`. Rejected — silent mismatch would produce unsorted/broken sweep results with no user-visible error until after trials run.

---

### R-004: Applying Per-Trial Hyperparameters to TrainingInfo

**Decision**: `SFTParameters.from_dict(dict(wandb.config))` constructs a new `SFTParameters` from the trial's hyperparameter values. `TrainingInfo` is immutable (Pydantic `frozen=True` is not set, but all fields are), so a new instance is constructed via `training_info.model_copy(update={"sft_parameters": new_sft_params})`. This new `TrainingInfo` is passed to `load_base_model` and `create_trainer` for the trial.

**Rationale**: `SFTParameters.from_dict()` already exists for this purpose. `model_copy(update=...)` is the Pydantic v2 idiomatic way to produce a modified copy of a model instance without mutation. No new infrastructure needed.

**Alternatives considered**: Mutating `training_info.sft_parameters` directly. Not possible — Pydantic model fields are not reassignable by default. Passing `sft_parameters` as a separate argument to helpers. Rejected — existing helpers accept `TrainingInfo` as the unit of training configuration; changing their signatures would violate the principle of minimal change.

---

### R-005: Per-Trial Output Directory Isolation (FR-007)

**Decision**: Each trial's SFTTrainer writes checkpoints to `outputs/<model_name>/<wandb_run_id>/`. `CommonPaths` gains a `sweep_trial_outputs(model_name, run_id)` method returning this path. `TrainingInfo.create_sft_config()` currently hardcodes `output_dir=self.model_name`; a new `create_sft_config_with_output_dir(dataset_info, report_to_wandb, output_dir)` overload (or a simple `output_dir` parameter default `None`) will allow the sweep to pass an isolated directory per trial.

**Rationale**: The existing `output_dir=self.model_name` in `TrainingInfo.create_sft_config()` would cause all trials to collide in the same directory. Trials run sequentially (single-agent) so collisions wouldn't corrupt checkpoints, but the last trial would overwrite all previous ones, making sweep artifact retrieval impossible.

**Alternatives considered**: Using a fixed `outputs/<model_name>/sweep/` directory and relying on sequential execution for safety. Rejected — harder to trace which checkpoint belongs to which wandb run. Using `wandb.run.id` as the subdirectory name makes the mapping unambiguous.

---

## Phase 1: Design & Contracts

### Data Model

See [data-model.md](./data-model.md).

### CLI Contracts

See [contracts/sweep-command.md](./contracts/sweep-command.md).

### Quickstart

See [quickstart.md](./quickstart.md).
