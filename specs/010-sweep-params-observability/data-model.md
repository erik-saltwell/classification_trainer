# Data Model: Sweep Parameter Simplification and Observability

## Entities

### SweepInfo (modified)
Pydantic model in `configuration/sweep_info.py`. Unchanged fields. Modified method:

- `to_wandb_sweep_config(metric?, metric_goal?) -> dict`: Signature loses `sft_parameters` argument. Body changes to iterate only `self.parameters` (the swept subset), not all SFT fields.

### SFTParameters (unchanged)
Pydantic model in `configuration/sft_parameters.py`. No field or method changes. Used as the base for trial parameter merging.

### TrainingInfo (unchanged)
`sft_parameters: SFTParameters` field is the base config for sweep trials. No structural changes.

## Data Flows

### Sweep Registration (simplified)

```
SweepInfo.parameters            → to_wandb_sweep_config()  → wandb sweep dict
(only swept fields)                (swept fields only,          (parameters block
                                    no unlisted defaults)         is small)
```

Old flow sent N_all_sft_fields entries. New flow sends N_swept_fields entries (≤ N_all_sft_fields).

### Trial Parameter Construction (new merge pattern)

```
training_info.sft_parameters    ─┐
  (base: all fields, defaults)    ├→ merged dict → SFTParameters → TrainingInfo copy
run.config                       ─┘
  (delta: only swept fields,
   wandb-sampled values)
```

Merge rule: wandb values override base values for swept fields; all other fields come from base unchanged.

### Per-Trial Observability Table

```
run.config                      → rows: one per wandb-provided field
  (swept fields only)
                                   columns:
                                     [0] param name         (key from run.config)
                                     [1] wandb value        (value from run.config)
                                     [2] actual trainer val  (value from merged sft_parameters)
```

## Validation Rules

### SweepInfo (unchanged from current)
- `parameters` must be non-empty (at least one swept field)
- Every key in `parameters` must match a field in `SFTParameters.model_fields`
- Grid method: no range parameters (min/max)
- Domain validators per field (lora_dropout 0-1, rank > 0, etc.)

### apply_trial_sft_parameters (changed)
- Merged dict is validated by `SFTParameters.model_validate` — Pydantic enforces all field types and constraints
- Unknown keys in `run.config` that are not `SFTParameters` fields will cause a Pydantic `ValidationError`
  (acceptable: a well-formed sweep config can only inject known fields, validated at sweep creation time)
