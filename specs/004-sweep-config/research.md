# Research: User-Configurable Sweep Parameters

**Feature**: 004-sweep-config | **Date**: 2026-03-06

## R1: wandb Sweep Config Parameter Formats

**Decision**: Use wandb's native parameter specification formats directly in the generated sweep config dict.

**Rationale**: wandb's sweep API accepts parameter specs in three forms that map directly to our three user-facing formats:
- `{"values": [v1, v2, ...]}` — discrete list (wandb samples uniformly)
- `{"distribution": "uniform", "min": X, "max": Y}` — continuous uniform range
- `{"distribution": "log_uniform_values", "min": X, "max": Y}` — continuous log-uniform range (for learning rate)
- `{"value": X}` — fixed constant (wandb uses this exact value every trial)

The user's YAML uses a simplified syntax (bare scalar, `{values: [...]}`, `{min: X, max: X}`), and the system translates to wandb's format. No custom sampling logic needed.

**Alternatives considered**:
- Custom sampling layer on top of wandb: Rejected — adds complexity with no benefit; wandb already handles all distribution types.

## R2: wandb Sweep Methods

**Decision**: Support `random`, `bayes`, and `grid` — the three methods wandb natively supports.

**Rationale**: These are the only three values wandb's `method` field accepts. No translation layer needed; the user's string is passed directly to wandb.

**Notes**:
- `grid` requires all parameters to use `{"values": [...]}` format (no continuous distributions). wandb will error if violated, but we validate first for a better error message.
- `bayes` works with both discrete and continuous parameters.

## R3: Opt-In Parameter Semantics (Non-Sweep Defaults)

**Decision**: When a `sweep` block with `parameters` is present, unlisted parameters are passed to wandb as `{"value": X}` using their `sft_parameters` value. This ensures wandb's config contains all SFT parameter keys (needed by `apply_trial_sft_parameters`).

**Rationale**: The existing `apply_trial_sft_parameters` function builds an `SFTParameters` from `wandb.config`, which must contain all fields. Passing unlisted params as fixed `{"value": X}` entries ensures completeness without sweeping them.

**Alternatives considered**:
- Only include swept params in wandb config, fill unlisted from `sft_parameters` at trial time: Would work but requires changes to `apply_trial_sft_parameters` and risks config/display mismatches in wandb dashboard.
- Include only swept params and merge in helper: More code changes, less transparent in wandb UI.

## R4: Backward Compatibility (No Sweep Block)

**Decision**: When `training_info` has no `sweep` block (`sweep` field is `None`), `build_sweep_config` falls back to the existing hardcoded default search space from `SFTParameters.get_default_sweep_config()`.

**Rationale**: Existing configs must work unchanged. The `get_default_sweep_config` method is preserved as a fallback and renamed to clarify its role as the legacy default.

## R5: Distribution Auto-Selection

**Decision**: When a user specifies `{min: X, max: Y}`:
- `learning_rate` → `log_uniform_values` distribution
- All other numeric params → `uniform` distribution

**Rationale**: Learning rate is conventionally searched on a log scale because the difference between 1e-5 and 1e-4 is as meaningful as between 1e-4 and 1e-3. Other numeric params (weight_decay, lora_dropout, warmup_ratio) are linear-scale.

**Configuration**: The log-uniform parameter list is hardcoded to `{"learning_rate"}`. This could be made configurable later but YAGNI applies.

## R6: Trial Counter Implementation

**Decision**: Add a mutable `_trial_number` counter to the `SweepCommand` that increments at the start of each `_run_trial()` call. Display as `"Trial {n} of {count}"` via `LoggingProtocol`.

**Rationale**: wandb's agent API doesn't expose trial index. A simple counter on the command object is the simplest approach. The counter tracks how many trials the *local agent* has run, which matches the user's `--count` semantics.

**Alternatives considered**:
- Query wandb API for sweep run count: Adds network dependency, may count runs from other agents, overcomplicated.
