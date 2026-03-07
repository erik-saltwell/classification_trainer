# Contract: Sweep Block (training_info YAML)

**Feature**: 004-sweep-config | **Date**: 2026-03-06

## Location

The `sweep` block is an optional top-level key in any `training_info/*.yaml` file.

## Schema

```yaml
# Optional sweep configuration for hyperparameter sweeps.
# Omit this entire block to use the default search space.
sweep:
  # Search method: "random" (default), "bayes", or "grid"
  method: random

  # Parameters to sweep. Only listed parameters are varied;
  # all others use the fixed value from sft_parameters above.
  # Each parameter supports three formats:
  #   1. Discrete list:    {values: [v1, v2, v3]}
  #   2. Continuous range:  {min: <lower>, max: <upper>}
  #   3. Fixed value:       <scalar>
  parameters:
    rank: {values: [8, 16, 32]}
    learning_rate: {min: 1e-5, max: 1e-3}
    optim: "adamw_bnb_8bit"
```

## Field Reference

### `sweep.method`

| Value | Description |
|-------|-------------|
| `random` | Random sampling from parameter distributions (default) |
| `bayes` | Bayesian optimization — converges faster for smooth landscapes |
| `grid` | Exhaustive grid search — all parameters must use discrete `{values: [...]}` format |

### `sweep.parameters`

Keys must be valid `sft_parameters` field names:
- `rank`, `alpha_multiplier`, `use_projection_modules`, `lora_dropout`
- `warmup_ratio`, `learning_rate`, `optim`, `weight_decay`, `lr_scheduler_type`

### Parameter Formats

**Discrete list** — sweep samples from these values:
```yaml
rank: {values: [8, 16, 32, 64]}
optim: {values: ["adamw_bnb_8bit", "adamw_torch"]}
```

**Continuous range** — sweep samples from a continuous distribution:
```yaml
learning_rate: {min: 1e-5, max: 1e-3}    # uses log-uniform distribution
weight_decay: {min: 0.0, max: 0.1}        # uses uniform distribution
```

Distribution is auto-selected: `learning_rate` uses log-uniform; all others use uniform.

**Fixed value** — parameter is held constant across all trials:
```yaml
optim: "adamw_bnb_8bit"
rank: 32
use_projection_modules: false
```

## Validation Errors (pre-trial)

| Condition | Error |
|-----------|-------|
| Invalid parameter name | `"Unknown sweep parameter 'X'. Valid: [rank, alpha_multiplier, ...]"` |
| Empty values list | `"Sweep parameter 'X' has empty values list"` |
| min >= max | `"Sweep parameter 'X': min (Y) must be less than max (Z)"` |
| grid + continuous range | `"Grid search requires all parameters to use discrete values. 'X' uses min/max range"` |
| No parameters | `"Sweep block requires at least one parameter in 'parameters'"` |

## Backward Compatibility

- **No `sweep` key**: Sweep command uses existing hardcoded default search space. No change in behavior.
- **`sweep` key present**: Full opt-in control. Only listed parameters are swept; unlisted use `sft_parameters` values.
- **Non-sweep commands** (`train`, `analyze-dataset`, etc.): `sweep` block is ignored entirely.

## Terminal Output Contract

At each trial start:
```
Trial 3 of 10
Trial parameters:
  rank: 32
  alpha_multiplier: 2
  learning_rate: 0.000142
  optim: adamw_bnb_8bit
  ...
```
