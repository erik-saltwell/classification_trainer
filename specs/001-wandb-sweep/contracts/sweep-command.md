# CLI Contract: sweep command

**Feature**: `001-wandb-sweep`
**Date**: 2026-03-05

## Command Signature

```
classification-trainer sweep
    --dataset       <name>          Dataset info YAML name (no extension)
    --base-model    <name>          Base model info YAML name (no extension)
    --training-info <name>          Training info YAML name (no extension)
    --inference-info <name>         Inference info YAML name (no extension)
    [--count        <int>]          Max number of trials for this agent (default: 10)
```

All four named config arguments are required. `--count` is optional.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Sweep agent ran to completion (all `--count` trials finished or sweep controller stopped it) |
| 1    | Startup validation failed (invalid config file, `sweep_metric` not in registry, missing `wandb_config` in training info) |

Individual trial failures (OOM, evaluation error) do NOT cause a non-zero exit code for the sweep command — they are marked as failed runs in wandb and the agent continues.

## Startup Validation (runs before any trial)

The command exits with code 1 and a descriptive error message if any of the following are true:

1. Any config YAML file is missing or fails Pydantic validation
2. `training_info.wandb_config` is `None`
3. `inference_info.sweep_metric` is not a key in `_METRIC_REGISTRY`
4. `inference_info.sweep_metric_goal` is not `"maximize"` or `"minimize"`

## Behaviour

1. Loads all four YAML configs
2. Runs startup validation (see above)
3. Calls `wandb.sweep()` with config from `SFTParameters.get_default_sweep_config(sweep_metric, sweep_metric_goal)`
4. Calls `wandb.agent(sweep_id, trial_fn, count=count)` — blocks until all trials complete
5. Each trial:
   a. Calls `wandb.init()` — fresh run, step counter starts at 0
   b. Reads `wandb.config` for this trial's `SFTParameters` values
   c. Builds a modified `TrainingInfo` with the trial's `SFTParameters`
   d. Loads model + tokenizer fresh
   e. Loads, preps, and splits dataset
   f. Trains via existing `create_trainer` / `run_training` helpers
   g. Evaluates: runs classifier inference on test set, computes all configured metrics
   h. Logs all metrics to wandb at `final_global_step + 1` (includes the sweep target metric)
   i. Calls `wandb.finish()`
   j. On any exception: calls `wandb.finish(exit_code=1)`, marks trial failed, continues

## Example Invocation

```bash
classification-trainer sweep \
    --dataset rpg_reddit_post_classification \
    --base-model qwen2.5-0.5b-instruct \
    --training-info reddit-rpg-rules-questions-classifier \
    --inference-info simple_classification \
    --count 20
```

## Wandb Output

After each trial completes, the wandb dashboard for the sweep project shows:
- All trial runs ranked by `sweep_metric` (e.g., `f1`)
- Per-trial hyperparameter values matching `sft_parameters` YAML field names exactly
- Failed trials marked as failed (not contributing to ranking)

The sweep ID is printed to stdout when the sweep is created, enabling additional agents to join the same sweep from other machines:
```
wandb agent --count 10 <entity>/<project>/<sweep_id>
```

## Relationship to Existing Commands

| Aspect | `train` command | `sweep` command |
|--------|----------------|-----------------|
| Config sources | Same 4 YAMLs + optional publishing | Same 4 YAMLs, no publishing |
| Wandb | Single run per invocation | One run per trial, multiple trials |
| SFTParameters | Fixed from training YAML | Varied per trial by sweep controller |
| Post-training eval | Logs pre + post metrics | Logs post-training metrics per trial |
| Output | Trained model (optionally published) | Sweep results in wandb dashboard |
