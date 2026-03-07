# Data Model: User-Configurable Sweep Parameters

**Feature**: 004-sweep-config | **Date**: 2026-03-06

## New Entities

### SweepParameterSpec

Represents a single parameter's sweep specification. Exactly one of the three formats must be provided.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `values` | `list[int \| float \| str \| bool]` | One of three | Discrete list of values to sample from. Must not be empty. |
| `min` | `float` | One of three | Lower bound of continuous range. Must be < `max`. |
| `max` | `float` | One of three | Upper bound of continuous range. Must be > `min`. |
| `value` | `int \| float \| str \| bool` | One of three | Fixed constant value (derived from bare scalar in YAML). |

**Validation rules**:
- Exactly one format: `values` list, `min`+`max` pair, or `value` scalar.
- `values` must contain at least one element.
- `min` must be strictly less than `max`.
- `min` and `max` must both be present or both absent.
- Values must be valid for the target `SFTParameters` field (e.g., `lora_dropout` values must be 0.0-1.0, `optim` values must be valid `OptimizerType` strings).

### SweepConfig

Top-level sweep configuration block. Optional on `TrainingInfo`.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `method` | `str` (enum: `random`, `bayes`, `grid`) | No | `random` | Search strategy for the sweep. |
| `parameters` | `dict[str, SweepParameterSpec]` | Yes | — | Map of `sft_parameters` field names to their sweep specifications. Must contain at least one entry. |

**Validation rules**:
- `parameters` must not be empty.
- All keys in `parameters` must be valid `SFTParameters` field names.
- If `method` is `grid`, no parameter may use `min`/`max` (continuous range) format.

## Modified Entities

### TrainingInfo

| Field | Change | Type | Default |
|-------|--------|------|---------|
| `sweep` | ADDED | `SweepConfig \| None` | `None` |

**Behavior**: When `sweep` is `None`, the sweep command uses the existing hardcoded default search space. When present, it controls the sweep search space with opt-in semantics.

### SFTParameters

| Method | Change | Notes |
|--------|--------|-------|
| `get_default_sweep_config()` | PRESERVED | Retained as fallback for backward compatibility when no `sweep` block is present. |

## Relationships

```text
TrainingInfo
├── sft_parameters: SFTParameters        (always present, provides fixed values for non-swept params)
└── sweep: SweepConfig | None            (optional, controls sweep behavior)
    ├── method: str                       (search strategy)
    └── parameters: dict[str, SweepParameterSpec]
        └── keys must be valid SFTParameters field names
```

## YAML ↔ wandb Config Translation

| User YAML Format | wandb Config Format |
|-------------------|---------------------|
| `rank: {values: [8, 16, 32]}` | `{"rank": {"values": [8, 16, 32]}}` |
| `learning_rate: {min: 1e-5, max: 1e-3}` | `{"learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3}}` |
| `weight_decay: {min: 0.0, max: 0.1}` | `{"weight_decay": {"distribution": "uniform", "min": 0.0, "max": 0.1}}` |
| `optim: "adamw_bnb_8bit"` | `{"optim": {"value": "adamw_bnb_8bit"}}` |
| `rank: 32` | `{"rank": {"value": 32}}` |
| *(param not listed)* | `{"<param>": {"value": <sft_parameters value>}}` |
