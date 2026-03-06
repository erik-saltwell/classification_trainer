# W&B Sweeps for Unsloth LLM Fine-Tuning (Python)

This tutorial shows a practical, **copy/paste-able** setup for running **Weights & Biases (W&B) Sweeps** to tune hyperparameters for **LLM supervised fine-tuning (SFT)** using **Unsloth + TRL’s `SFTTrainer`**.

You’ll end up with:
- A single training script (`train_sweep.py`) that can run **one run** or be driven by a **sweep agent**
- A sweep configuration (Python dict, plus an optional YAML)
- Commands for running **single-machine** and **multi-machine** sweeps

---

## 1) How sweeps work (30 seconds)

A W&B sweep has:
- A **sweep configuration**: what to vary, what metric to optimize
- A **controller** (usually hosted by W&B) that decides the next hyperparameter set
- One or more **agents** (your machines) that repeatedly run your training function, once per trial

Each trial is a normal W&B run. Sweeps can run **grid**, **random**, or **Bayesian** search.

---

## 2) Prerequisites

### Accounts / auth
- A W&B account (free is fine)
- Logged in on the machine(s) that will run the sweep agent(s)

```bash
pip install -U wandb
wandb login
```

### Python environment
Install your stack. Exact versions vary; this set is typical:

```bash
pip install -U unsloth trl transformers datasets accelerate peft bitsandbytes
```

> If you’re using Unsloth’s Docker image or a Colab/Kaggle notebook, you may already have most of this.

---

## 3) Pick a “sweep-friendly” training setup

Sweeps are expensive for LLMs. A good sweep setup:
- Uses a **small-ish dataset slice** and **`max_steps`** (not full epochs) at first
- Keeps your objective metric meaningful (usually `eval_loss` first)
- Avoids searching too many “structural” knobs at once (like rank + max_seq_length + optimizer + schedulers)

Recommended starting ranges for LoRA/QLoRA are often:
- `learning_rate` around `2e-4` down to `5e-6`
- 1–3 epochs (or a modest `max_steps`)
- A stable effective batch size via `micro_batch * grad_accum`

---

## 4) The training script (works with sweeps)

Save this as **`train_sweep.py`**.

It:
- Uses **Unsloth** to load a model + apply LoRA
- Uses **TRL `SFTTrainer`** for training + evaluation
- Uses **W&B Sweeps** via `wandb.sweep()` + `wandb.agent()`
- Logs `eval_loss` explicitly at the end to make sweep optimization robust

```python
import argparse
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import wandb
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


# ----------------------------
# Configuration you WILL edit
# ----------------------------

DEFAULT_MODEL_NAME = "unsloth/llama-3.1-8b-unsloth-bnb-4bit"  # change to your base model
DEFAULT_DATASET = "roneneldan/TinyStories"                   # simple 'text' dataset (good for a smoke-test)
DEFAULT_DATASET_SPLIT = "train[:1%]"                         # tiny slice so sweeps are cheap
DEFAULT_TEXT_FIELD = "text"

# LoRA target modules commonly used in Unsloth examples
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ----------------------------
# Utilities
# ----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_dtype() -> torch.dtype:
    # Unsloth helper: prefer bf16 when supported
    return torch.bfloat16 if is_bfloat16_supported() else torch.float16


def build_model_and_tokenizer(cfg: wandb.sdk.wandb_config.Config) -> tuple[Any, Any]:
    # Create a *fresh* model/tokenizer for each sweep trial.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        dtype=pick_dtype(),
        load_in_4bit=cfg.load_in_4bit,
        trust_remote_code=cfg.trust_remote_code,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        use_gradient_checkpointing=cfg.use_gradient_checkpointing,
        random_state=cfg.seed,
        max_seq_length=cfg.max_seq_length,
    )
    return model, tokenizer


def load_train_eval_datasets(
    dataset_name: str,
    split: str,
    text_field: str,
    seed: int,
    eval_fraction: float,
):
    ds = load_dataset(dataset_name, split=split)

    # Minimal sanity check: ensure expected column exists
    if text_field not in ds.column_names:
        raise ValueError(
            f"Dataset '{dataset_name}' split '{split}' has columns {ds.column_names}, "
            f"but text_field='{text_field}' was not found."
        )

    # Make an eval split for sweep optimization
    split_ds = ds.train_test_split(test_size=eval_fraction, seed=seed)
    return split_ds["train"], split_ds["test"]


# ----------------------------
# The sweep training function
# ----------------------------

def train_one_trial(project: str, entity: Optional[str]) -> None:
    """One sweep trial.

    IMPORTANT:
    - The sweep controller sets wandb.config for this run.
    - This function must call wandb.init() itself.
    """
    run = wandb.init(project=project, entity=entity)
    cfg = wandb.config

    # Keep outputs isolated per trial
    run_dir = os.path.join(cfg.output_root, run.id)

    seed_everything(cfg.seed)

    train_ds, eval_ds = load_train_eval_datasets(
        dataset_name=cfg.dataset_name,
        split=cfg.dataset_split,
        text_field=cfg.text_field,
        seed=cfg.seed,
        eval_fraction=cfg.eval_fraction,
    )

    model, tokenizer = build_model_and_tokenizer(cfg)

    # You can sweep warmup_ratio and convert it to steps
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)

    # NOTE: TRL uses SFTConfig (TrainingArguments-like).
    # To ensure metrics flow to W&B, set report_to=["wandb"].
    sft_args = SFTConfig(
        output_dir=run_dir,
        max_seq_length=cfg.max_seq_length,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"],
        run_name=run.name,  # surfaces in HF logs + W&B
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
    )

    # TRL v0.29+ calls this parameter `processing_class`;
    # some examples use `tokenizer=`.
    # If your TRL version errors, replace processing_class=tokenizer with tokenizer=tokenizer.
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # Ensure eval metrics exist for the sweep
    eval_metrics = trainer.evaluate()
    # Most HF/TRL trainers report eval loss under "eval_loss"
    if "eval_loss" in eval_metrics:
        wandb.log({"eval_loss": eval_metrics["eval_loss"]})

    run.finish()


# ----------------------------
# Sweep configuration
# ----------------------------

def make_sweep_config() -> Dict[str, Any]:
    # Keep the search space modest at first. Expand once your pipeline is stable.
    return {
        "name": "unsloth-sft-sweep",
        "method": "bayes",  # try "random" first if you're new to sweeps
        "metric": {"name": "eval_loss", "goal": "minimize"},
        "run_cap": 20,  # optional: global cap for the sweep (still use agent --count for local caps)
        "parameters": {
            # Dataset/model knobs (usually keep fixed during sweeps)
            "model_name": {"value": DEFAULT_MODEL_NAME},
            "dataset_name": {"value": DEFAULT_DATASET},
            "dataset_split": {"value": DEFAULT_DATASET_SPLIT},
            "text_field": {"value": DEFAULT_TEXT_FIELD},

            # Repro / housekeeping
            "seed": {"value": 3407},
            "output_root": {"value": "outputs"},
            "eval_fraction": {"value": 0.01},
            "trust_remote_code": {"value": False},

            # Unsloth loading
            "max_seq_length": {"value": 2048},
            "load_in_4bit": {"value": True},

            # LoRA knobs (structural): keep this set small initially
            "lora_r": {"values": [8, 16, 32]},
            "lora_alpha": {"values": [16, 32, 64]},
            "lora_dropout": {"values": [0.0, 0.05]},
            "target_modules": {"value": DEFAULT_TARGET_MODULES},
            "use_gradient_checkpointing": {"values": ["unsloth", True]},

            # Training knobs (usually most important)
            "learning_rate": {"values": [2e-4, 1e-4, 5e-5, 2e-5]},
            "weight_decay": {"values": [0.0, 0.01, 0.05]},
            "micro_batch_size": {"values": [1, 2, 4]},
            "grad_accum_steps": {"values": [4, 8, 16]},

            # Schedule / speed controls (critical for “cheap” sweeps)
            "max_steps": {"values": [50, 100, 200]},
            "warmup_ratio": {"values": [0.03, 0.05, 0.1]},

            # Logging / eval cadence
            "logging_steps": {"value": 1},
            "eval_steps": {"values": [10, 25]},
            "save_steps": {"values": [25, 50]},
            "save_total_limit": {"value": 2},
        },
    }


# ----------------------------
# Entry point
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "unsloth-sweeps"))
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--count", type=int, default=10, help="Max number of sweep trials for this agent.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sweep_config = make_sweep_config()

    # IMPORTANT: sweep + runs must be in the same project.
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)

    # Start an agent locally (one machine). For multi-machine sweeps, see below.
    wandb.agent(
        sweep_id=sweep_id,
        function=lambda: train_one_trial(project=args.project, entity=args.entity),
        count=args.count,
    )


if __name__ == "__main__":
    # Required if you ever run with multiprocessing / torchrun
    main()
```

---

## 5) Run the sweep (single machine)

```bash
python train_sweep.py --project my-unsloth-sweep --count 10
```

That will:
1) create the sweep
2) start one local agent
3) run up to `--count` trials

---

## 6) Parallelize the sweep (multiple agents / machines)

**Best practice:** initialize the sweep once, then run many agents.

### Option A: Initialize in Python, run agents elsewhere
1) Run once to create a sweep ID (you can modify the script to “create only”)
2) On each machine, run:

```bash
wandb agent --count 5 ENTITY/PROJECT/SWEEP_ID
```

### Option B: Use a YAML sweep (CLI-friendly)

Create **`sweep.yaml`**:

```yaml
program: train_sweep.py
name: unsloth-sft-sweep
method: bayes
metric:
  name: eval_loss
  goal: minimize
run_cap: 20
parameters:
  learning_rate:
    values: [0.0002, 0.0001, 0.00005, 0.00002]
  micro_batch_size:
    values: [1, 2, 4]
  grad_accum_steps:
    values: [4, 8, 16]
  lora_r:
    values: [8, 16, 32]
  lora_alpha:
    values: [16, 32, 64]
```

Initialize the sweep:

```bash
wandb sweep --project my-unsloth-sweep sweep.yaml
```

Start one or more agents (on the same box or different boxes):

```bash
wandb agent --count 10 ENTITY/my-unsloth-sweep/SWEEP_ID
```

---

## 7) What to optimize (beyond `eval_loss`)

`eval_loss` is a great first sweep objective because it’s cheap and always available.

Once stable, you can sweep toward task metrics:
- classification: F1 / accuracy
- instruction tuning: a small evaluation set scored by a lightweight judge / rubric
- generation: exact match / BLEU / ROUGE (task-dependent)

If you use a custom metric, make sure:
- the metric name in the sweep config matches what you log (exact string)
- it’s logged at the top level (not nested)

---

## 8) Practical sweep tips for LLM fine-tuning

- **Keep runs cheap** at first (`max_steps` + dataset slice). If your training pipeline is flaky, your sweep will be flaky.
- **Don’t sweep everything at once.** Start with LR + effective batch size (micro batch + grad accum).
- **Log “final eval metric” explicitly** after `trainer.evaluate()` to make sweeps reliable.
- **Use `run_name` and structured output dirs** so checkpoints don’t collide across trials.
- **Cap the sweep**: Bayesian and random searches will otherwise keep going unless you stop the agent.

---

## 9) Troubleshooting

### “My sweep runs forever”
- For random/bayes, this is expected unless you cap it.
- Use `wandb agent --count N ...` or `wandb.agent(..., count=N)`.

### “Sweep says it can’t find my metric”
- Make sure the metric name matches, e.g. `"eval_loss"`.
- Ensure it is logged directly: `wandb.log({"eval_loss": value})`.

### “Multiprocessing / torchrun launches multiple sweeps”
- Put `wandb.sweep()` and `wandb.agent()` behind:
  ```python
  if __name__ == "__main__":
      main()
  ```

### “W&B run finishes before I log post-training metrics”
- Log them *before* `run.finish()` (or before exiting the `wandb.init()` context manager).

---

## References (URLs in code blocks)

W&B Sweeps docs:
```text
https://docs.wandb.ai/models/sweeps
https://docs.wandb.ai/models/sweeps/define-sweep-configuration
https://docs.wandb.ai/models/sweeps/initialize-sweeps
https://docs.wandb.ai/models/sweeps/start-sweep-agents
https://docs.wandb.ai/models/sweeps/add-w-and-b-to-your-code
```

W&B + Hugging Face integration:
```text
https://docs.wandb.ai/models/integrations/huggingface
```

Unsloth docs + examples:
```text
https://github.com/unslothai/unsloth
https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint
https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
```

TRL SFTTrainer docs:
```text
https://huggingface.co/docs/trl/en/sft_trainer
```
