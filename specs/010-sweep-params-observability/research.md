# Research: Sweep Parameter Simplification and Observability

## Decision 1: wandb sweep config with only swept parameters

**Decision**: Send only the swept parameters in the wandb sweep definition. Do not include fixed-value entries for unlisted fields.

**Rationale**: wandb's `wandb.sweep()` API accepts a `parameters` dict where each key names a variable. Parameters omitted from this dict simply do not appear in `run.config` on the agent side. This means the agent receives only the values wandb sampled — exactly the values that differ from the base config. The base defaults are already in `training_info.sft_parameters` and are applied by merging.

**Alternatives considered**:
- Keep sending all fields as fixed values for unlisted params (current approach). Rejected: redundant with `sft_parameters`, creates drift risk, clutters wandb UI with fixed "swept" parameters that never vary.

## Decision 2: Apply trial config as an overlay on base SFTParameters

**Decision**: `apply_trial_sft_parameters` merges `training_info.sft_parameters.to_dict()` with `dict(run.config)`, with wandb values winning on conflict. Then constructs `SFTParameters` from the merged dict.

**Rationale**: wandb now only sends swept fields. The base dict fills in all unswept fields with their training_info values. The merge is a simple dict union — no new abstraction needed.

**Alternatives considered**:
- Construct `SFTParameters` purely from `run.config`. Rejected: fails when wandb only sends swept subset — missing fields cause Pydantic validation errors.
- Have wandb send all fields (old approach). Rejected: this is the problem being solved.

## Decision 3: Logging the sweep config at initialization

**Decision**: Log the sweep parameters dict using `logger.report_message()` with `json.dumps(... indent=2)` immediately after `generate_sweep_parameters()` in `execute()`, before calling `initialize_sweep()`.

**Rationale**: `report_message` with formatted JSON gives a human-readable view of the full sweep definition. The sweep config is a nested dict (method, metric, parameters), not a flat table — so `report_multicolumn_table` is not the right fit here.

**Alternatives considered**:
- `report_table_message` (flat key-value). Rejected: the config is nested; a flat table would either truncate or be confusing.
- Log after `initialize_sweep`. Rejected: if `wandb.sweep()` fails, the researcher still needs to see what was sent to diagnose the problem.

## Decision 4: Per-trial parameter table — when and where

**Decision**: Log the table in `SweepCommand.run_single_trial()` immediately after `apply_trial_sft_parameters()` sets `self.runner.training_info`. Rows cover only the fields present in `run.config` (the swept fields). The "actual trainer value" is read from `self.runner.training_info.sft_parameters`.

**Rationale**: After `apply_trial_sft_parameters`, `self.runner.training_info.sft_parameters` reflects exactly what the trainer will use — this is the ground truth. The trainer is deterministically constructed from those values, so logging here is equivalent to logging "after trainer creation". This keeps the logging in the command layer (sweep.py), consistent with Principle III.

**Alternatives considered**:
- Log inside `training_helper.create_trainer`. Rejected: helpers must not contain orchestration logic (Principle III). The trainer helper has no access to `run.config`.
- Log after `train_model` returns. Rejected: too late — the trial is already over. The point is to confirm config before training runs.
- Pass a callback into `train_model`. Rejected: unnecessary complexity for what is a command-layer concern.

## Decision 5: Removing sft_parameters from to_wandb_sweep_config

**Decision**: Remove the `sft_parameters: SFTParameters` argument from `SweepInfo.to_wandb_sweep_config()`. The method now only converts `self.parameters` (swept fields) to wandb format — it has no need for defaults.

**Rationale**: The only reason `sft_parameters` was passed in was to populate fixed-value entries for unlisted fields. With Decision 1 (only swept fields in wandb config), that population is gone. The argument becomes dead weight.

**Impact**: The optional `metric` and `metric_goal` parameters added during the last bug-fix session become the only parameters. Call sites in `sweep.py` drop the `sft_parameters` argument. Tests updated accordingly.
