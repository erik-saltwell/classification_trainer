# Implementation Plan: Sweep Parameter Simplification and Observability

**Branch**: `010-sweep-params-observability` | **Date**: 2026-03-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/010-sweep-params-observability/spec.md`

## Summary

Remove the requirement that every `SFTParameters` field be listed in `sweep_config.parameters`. The sweep config only names the fields being varied; wandb receives only those fields. At trial time, wandb's sampled values are merged onto the base `sft_parameters` from `training_info`. Add logging at sweep initialization (full sweep config) and at the start of each trial (per-parameter comparison table).

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Pydantic v2, wandb sweep/agent API, TRL SFTTrainer, Typer/Rich
**Storage**: N/A (config files only — YAML in `training_info/`)
**Testing**: pytest (`tests/unit/`)
**Target Platform**: Linux (GPU training host)
**Project Type**: CLI tool / training library
**Performance Goals**: No impact — changes are config-layer and logging only; no training path is altered
**Constraints**: Must satisfy pyright, mypy, ruff. No new dependencies.
**Scale/Scope**: Affects `sweep_info.py`, `sweep_helper.py`, `sweep.py`, example YAML, and unit tests.

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-First | ✅ Pass | YAML still loads through Pydantic. `SweepInfo.parameters` remains the validated config block. No hardcoded values introduced. No new directories — `CommonPaths` unchanged. |
| II. Protocol-Based Interfaces | ✅ Pass | All logging goes through `LoggingProtocol`. No new concrete types at boundaries. |
| III. Separation of Concerns | ✅ Pass | Logging added in `sweep.py` (command layer). Parameter merging stays in `sweep_helper.py` (helper). `to_wandb_sweep_config` stays in `sweep_info.py` (configuration). |
| IV. Observability | ✅ Pass | Adds sweep-init and per-trial logging. WandB step reporting unchanged. |
| V. Simplicity & Scope | ✅ Pass | Net reduction in complexity: shorter `to_wandb_sweep_config`, simpler YAML. No new abstractions. |

## Project Structure

### Documentation (this feature)

```text
specs/010-sweep-params-observability/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research decisions
├── data-model.md        # Phase 1 data model
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (files touched)

```text
src/classification_trainer/
├── configuration/
│   └── sweep_info.py            # Remove sft_parameters arg from to_wandb_sweep_config
├── helpers/
│   └── sweep_helper.py          # Fix apply_trial_sft_parameters to merge onto base
└── commands/
    └── sweep.py                 # Add sweep-init log; add per-trial param table

training_info/
└── example.yaml                 # Update sweep_config comment — only swept fields needed

tests/unit/
└── test_sweep_info.py           # Update to_wandb_sweep_config call sites; add merge tests
```

**Structure Decision**: Single project, existing layout. No new files, no new directories.

## Changes

### Change 1 — `configuration/sweep_info.py`: Simplify `to_wandb_sweep_config`

**File**: `src/classification_trainer/configuration/sweep_info.py`

Remove the `sft_parameters: SFTParameters` parameter. The body no longer iterates all SFT fields — it only converts `self.parameters` (the swept subset).

**Before** (signature + body excerpt):
```python
def to_wandb_sweep_config(
    self,
    sft_parameters: SFTParameters,
    metric: str | None = None,
    metric_goal: str | None = None,
) -> dict[str, Any]:
    resolved_metric = metric if metric is not None else self.metric
    resolved_goal = metric_goal if metric_goal is not None else self.metric_goal
    sft_dict = sft_parameters.to_dict()
    wandb_params: dict[str, Any] = {}
    for field_name in SFTParameters.model_fields:            # iterates ALL fields
        if field_name in self.parameters:
            wandb_params[field_name] = self.parameters[field_name].to_wandb_param_spec(field_name)
        else:
            wandb_params[field_name] = {"value": sft_dict[field_name]}  # fixed values for unlisted
    return {
        "name": self.sweep_name,
        ...
        "parameters": wandb_params,
    }
```

**After**:
```python
def to_wandb_sweep_config(
    self,
    metric: str | None = None,
    metric_goal: str | None = None,
) -> dict[str, Any]:
    resolved_metric = metric if metric is not None else self.metric
    resolved_goal = metric_goal if metric_goal is not None else self.metric_goal
    wandb_params: dict[str, Any] = {
        field_name: spec.to_wandb_param_spec(field_name)
        for field_name, spec in self.parameters.items()   # only swept fields
    }
    return {
        "name": self.sweep_name,
        "description": self.description,
        "run_cap": self.run_cap,
        "method": str(self.method),
        "metric": {"goal": resolved_goal, "name": resolved_metric},
        "parameters": wandb_params,
    }
```

Also remove the now-unused `SFTParameters` import at the top of `sweep_info.py` (was only needed to iterate `model_fields` and call `to_dict()`). `LRSchedulerType` and `OptimizerType` are still needed for `_DOMAIN_VALIDATORS`.

### Change 2 — `helpers/sweep_helper.py`: Merge wandb config onto base SFTParameters

**File**: `src/classification_trainer/helpers/sweep_helper.py`

**Before**:
```python
def apply_trial_sft_parameters(
    training_info: TrainingInfo,
    trial_config: Mapping[str, Any],
) -> TrainingInfo:
    sft_params = SFTParameters.from_dict(dict(trial_config))   # fails if only swept fields present
    return training_info.model_copy(update={"sft_parameters": sft_params})
```

**After**:
```python
def apply_trial_sft_parameters(
    training_info: TrainingInfo,
    trial_config: Mapping[str, Any],
) -> TrainingInfo:
    base_dict = dict(training_info.sft_parameters.to_dict())
    merged = {**base_dict, **dict(trial_config)}               # wandb values override base
    sft_params = SFTParameters.from_dict(merged)
    return training_info.model_copy(update={"sft_parameters": sft_params})
```

### Change 3 — `commands/sweep.py`: Logging

**File**: `src/classification_trainer/commands/sweep.py`

Three sub-changes:

**3a. Update `generate_sweep_parameters`** — drop the `sft_parameters` argument:

```python
def generate_sweep_parameters(self) -> dict[str, Any]:
    assert self.training_info.sweep_config is not None
    return self.training_info.sweep_config.to_wandb_sweep_config()   # no sft_parameters arg
```

**3b. Log sweep config in `execute()`** — after `generate_sweep_parameters`, before `initialize_sweep`:

```python
parameters: dict[str, Any] = self.generate_sweep_parameters()
self.logger.report_message("[blue]Sweep config submitted to wandb:[/blue]")
self.logger.report_message(json.dumps(parameters, indent=2))
sweep_id: str = self.initialize_sweep(...)
```

Add `import json` at top of file.

**3c. Log per-trial parameter table in `run_single_trial()`** — after `apply_trial_sft_parameters`, before `load_model`:

```python
def run_single_trial(self) -> None:
    with initialize_wandb(self.training_info, self.dataset_info, WandBJobType.SWEEP) as run:
        sweep_run_config: dict[str, Any] = dict(run.config)
        self.runner.training_info = apply_trial_sft_parameters(self.training_info, sweep_run_config)

        # Log per-trial parameter comparison table
        trial_sft = self.runner.training_info.sft_parameters.to_dict()
        rows = [
            [name, str(wandb_val), str(trial_sft.get(name, "N/A"))]
            for name, wandb_val in sweep_run_config.items()
        ]
        self.logger.report_message("[blue]Trial sweep parameters:[/blue]")
        self.logger.report_multicolumn_table(
            headers=["Parameter", "WandB Value", "Trainer Value"],
            rows=rows,
        )

        self.runner.load_model(self.logger)
        current_step: int = self.runner.train_model(self.logger)
        results: list[MetricResult] = self.runner.evaluate_model(self.logger, F1Metric())
        self.reporter.report(results, current_step + 1)
```

### Change 4 — `training_info/example.yaml`: Update sweep_config comment

Update the comment in the `sweep_config` example block to clarify that `parameters` only lists fields to be varied:

```yaml
#   # Parameters to sweep. Each key must be a valid sft_parameters field name.
#   # Only list the parameters you want to vary — omitted parameters use their
#   # values from the sft_parameters block above (no need to repeat them here).
```

Remove the "3. Fixed value" format from the documentation comment since that use case is now better served by just not listing the parameter (it will use the base value). Fixed `value:` entries in the spec remain valid Pydantic-wise but should be discouraged in docs.

### Change 5 — `tests/unit/test_sweep_info.py`: Update tests

**5a. Update all `to_wandb_sweep_config(sft, ...)` calls** — remove the `sft` positional argument:

Every call of the form `sweep_info.to_wandb_sweep_config(sft_parameters, ...)` becomes `sweep_info.to_wandb_sweep_config(...)`.

**5b. Update assertions about wandb_params content** — the output `parameters` dict now only contains swept fields, not all SFT fields. Tests that assert on the full set of keys in the output must be updated to assert only on the swept fields.

**5c. Add tests for `apply_trial_sft_parameters` with partial config** (new tests in `test_sweep_info.py` or a new `test_sweep_helper.py`):

```python
def test_apply_trial_merges_onto_base():
    # Only rank is swept — other fields come from base
    base_info = ...  # TrainingInfo with default sft_parameters (rank=16, learning_rate=2e-4)
    trial_config = {"rank": 32}
    result = apply_trial_sft_parameters(base_info, trial_config)
    assert result.sft_parameters.rank == 32            # wandb overrides
    assert result.sft_parameters.learning_rate == 2e-4  # base value preserved

def test_apply_trial_all_swept_fields_override():
    # Multiple fields swept
    ...

def test_apply_trial_empty_config_returns_base():
    # Empty trial config → base unchanged
    ...
```

## Complexity Tracking

No constitution violations. No complexity justification required.
