# Data Model: WandB Hyperparameter Sweep

**Feature**: `001-wandb-sweep`
**Date**: 2026-03-05

## Modified Models

### InferenceInfo (configuration/inference_info.py)

Two new optional fields added to the existing Pydantic model:

| Field | Type | Default | Validation |
|-------|------|---------|------------|
| `sweep_metric` | `str` | `"f1"` | Must be a key in `_METRIC_REGISTRY`: `accuracy`, `precision`, `recall`, `f1`, `total_seen` |
| `sweep_metric_goal` | `str` | `"maximize"` | Must be `"maximize"` or `"minimize"` |

Validation of `sweep_metric` against `_METRIC_REGISTRY` happens at `SweepCommand.execute()` startup, not in the Pydantic model itself (to keep the model a pure data container per Constitution I).

**YAML example** (`inference_info/simple_classification.yaml` after change):
```yaml
do_sample: false
temperature: 0.0
top_p: 1.0
max_new_tokens: 8
repetition_penalty: null
prepare_unsloth_inference: true
metrics: ["accuracy", "precision", "recall", "f1", "total_seen"]
sweep_metric: "f1"
sweep_metric_goal: "maximize"
```

Both fields are optional — existing inference YAML files remain valid without them.

---

### CommonPaths (utils/common_paths.py)

One new method added:

| Method | Signature | Returns | Created by ensure_all_dirs_exist? |
|--------|-----------|---------|-----------------------------------|
| `sweep_trial_outputs` | `(model_name: str, run_id: str) -> Path` | `outputs/<model_name>/<run_id>/` | No (runtime output) |

This method returns `CommonPaths.OUTPUTS_DIR / model_name / run_id`. It is a runtime output directory and MUST NOT be added to `ensure_all_dirs_exist()` per Constitution I.

---

### TrainingInfo (configuration/training_info.py)

One new optional parameter added to `create_sft_config`:

| Method | New Parameter | Type | Default | Notes |
|--------|--------------|------|---------|-------|
| `create_sft_config` | `output_dir` | `str \| None` | `None` | When `None`, falls back to `self.model_name` (existing behaviour unchanged). When provided, overrides `output_dir` in `SFTConfig`. |

This is a backwards-compatible change — all existing callers pass no `output_dir` argument and behaviour is identical.

---

## New Entities

### SweepCommand (commands/sweep.py)

Dataclass implementing `CommmandProtocol`. Fields:

| Field | Type | Notes |
|-------|------|-------|
| `dataset_info` | `DatasetInfo` | Loaded from YAML |
| `base_model_info` | `BaseModelInfo` | Loaded from YAML |
| `training_info` | `TrainingInfo` | Loaded from YAML; `sft_parameters` is replaced per-trial |
| `inference_info` | `InferenceInfo` | Contains `sweep_metric` and `sweep_metric_goal` |
| `count` | `int` | Max number of trials for this agent; default `10` |

---

## New Helpers (helpers/sweep_helper.py)

Three pure functions, no state:

| Function | Inputs | Output | Notes |
|----------|--------|--------|-------|
| `build_sweep_config` | `inference_info: InferenceInfo` | `dict` | Calls `SFTParameters.get_default_sweep_config(inference_info.sweep_metric, inference_info.sweep_metric_goal)` |
| `apply_trial_sft_parameters` | `training_info: TrainingInfo, trial_config: Mapping[str, Any]` | `TrainingInfo` | Extracts SFTParameters fields from `trial_config`, constructs `SFTParameters.from_dict(...)`, returns `training_info.model_copy(update={"sft_parameters": ...})` |
| `extract_target_metric` | `results: list[MetricResult], metric_name: str` | `float` | Finds the `MetricResult` whose `metric_name` matches; raises `ValueError` if not found |

---

## State Transitions

Each sweep trial follows this lifecycle:

```
[PENDING]
    ↓ wandb.agent calls trial_fn
[INITIALIZING] — wandb.init(), load configs
    ↓
[TRAINING] — load_base_model, create_trainer, run_training
    ↓
[EVALUATING] — setup_unsloth_inference, add_inferred_column, generate_metrics
    ↓ WandBMetricsReporter.report(results, step=final_step+1)
[LOGGING] — target metric logged to wandb
    ↓ wandb.finish()
[COMPLETED]

[TRAINING / EVALUATING] → exception → wandb.finish(exit_code=1) → [FAILED]
```

Failed trials do not halt the sweep agent; the agent loop continues to the next trial.
