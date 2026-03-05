# Quickstart: WandB Hyperparameter Sweep

**Feature**: `001-wandb-sweep`
**Date**: 2026-03-05

## Prerequisites

- wandb account and `wandb login` completed on the machine
- `wandb_config` block present in your training info YAML
- Existing `inference_info` YAML (the new `sweep_metric` / `sweep_metric_goal` fields are optional and default to `f1` / `maximize`)

## Step 1: Verify your training info has wandb_config

```yaml
# training_info/my-classifier.yaml
model_name: "my-classifier"
hugging_face_user_name: "myuser"
# ... other fields ...
wandb_config:
  project: "my-classifier-sweeps"
  job_type: "sweep-trial"
```

## Step 2: (Optional) Set the sweep metric in inference_info

```yaml
# inference_info/simple_classification.yaml
metrics: ["accuracy", "precision", "recall", "f1", "total_seen"]
sweep_metric: "f1"          # which metric the sweep optimises for
sweep_metric_goal: "maximize"
```

If omitted, the sweep defaults to optimising for `f1` (maximize).

## Step 3: Run the sweep

```bash
classification-trainer sweep \
    --dataset my-dataset \
    --base-model qwen2.5-0.5b-instruct \
    --training-info my-classifier \
    --inference-info simple_classification \
    --count 20
```

The command will:
1. Print the wandb sweep URL
2. Run 20 training trials back-to-back
3. Each trial uses a different set of LoRA / SFT hyperparameters
4. Each trial is evaluated on the test set; F1 (or your chosen metric) is logged to wandb

## Step 4: Review results in wandb

Open the sweep URL printed at startup. The wandb sweep dashboard shows:
- All trials ranked by your target metric
- Per-trial hyperparameter values (named exactly as in the `sft_parameters` YAML block)
- Parallel coordinates plot for identifying which parameters matter most

## Step 5: Apply the best configuration

Find the best trial in wandb. Copy its hyperparameter values into a new or updated training YAML:

```yaml
# training_info/my-classifier-best.yaml
# ... same as before, but update sft_parameters:
sft_parameters:
  rank: 32                  # ← from best sweep trial
  alpha_multiplier: 2
  use_projection_modules: true
  lora_dropout: 0.03
  warmup_ratio: 0.05
  learning_rate: 0.00012
  optim: "adamw_bnb_8bit"
  weight_decay: 0.04
  lr_scheduler_type: "cosine"
```

Then run a full training + publish with those settings:

```bash
classification-trainer train \
    --dataset my-dataset \
    --base-model qwen2.5-0.5b-instruct \
    --training-info my-classifier-best \
    --inference-info simple_classification \
    --publishing-info my-publisher
```

## Running additional agents (multi-machine)

The sweep ID is printed when the sweep starts. To add more agents on other machines:

```bash
wandb agent --count 10 <entity>/<project>/<sweep_id>
```

## Troubleshooting

**"wandb_config must be set" error at startup**
Add a `wandb_config` block to your training info YAML (see Step 1).

**"Unknown metric" error at startup**
The value of `sweep_metric` must be one of: `accuracy`, `precision`, `recall`, `f1`, `total_seen`.

**A trial fails with OOM**
The trial is marked failed in wandb; the sweep continues. Consider reducing `per_device_batch_size` or `max_sequence_length` in your training info YAML, or reducing the search space for `rank` in a custom sweep config.

**Sweep runs indefinitely**
The `--count` argument caps the number of trials per agent. Without it, the agent defaults to 10 trials.
