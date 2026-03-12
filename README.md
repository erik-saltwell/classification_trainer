# classification-trainer

Fine-tune LLMs for binary text classification using [Unsloth](https://github.com/unslothai/unsloth), LoRA, and SFT (Supervised Fine-Tuning). Train a small instruct model to classify text into two categories, evaluate it, run hyperparameter sweeps with wandb, and publish the result to HuggingFace Hub.

## Prerequisites

- **Python 3.11+**
- **CUDA-capable GPU** with sufficient VRAM (8GB+ recommended for 0.5B–3B models)
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **[Weights & Biases](https://wandb.ai/) account** for experiment tracking and hyperparameter sweeps
- **[HuggingFace](https://huggingface.co/) account + access token** for downloading models and publishing results

## Installation

```bash
git clone <repo-url>
cd classification_trainer
uv sync
```

### Environment Variables

The following environment variables are required:

| Variable | Description |
|---|---|
| `HF_TOKEN` | [HuggingFace](https://huggingface.co/) access token for downloading models and publishing results |
| `WANDB_API_KEY` | [Weights & Biases](https://wandb.ai/) API key for experiment tracking and sweeps |
| `LOG_LEVEL` | Python log level (e.g., `WARNING`, `INFO`, `DEBUG`) — optional, defaults to `WARNING` |

You can set these in a `.env` file in the project root (see `.env.example`):

```
LOG_LEVEL=WARNING
WANDB_API_KEY=<your-api-key>
HF_TOKEN=<your-api-key>
```

## Quick Start

1. Create a dataset config at `dataset_info/my-dataset.yaml` (copy from `dataset_info/example.yaml`)
2. Create a training config at `training_info/my-model.yaml` (copy from `training_info/example.yaml`)
3. Create a base model config at `base_model_info/qwen2.5-0.5b-instruct.yaml` (copy from `base_model_info/example.yaml`)
4. Create a system prompt at `fragments/my_prompt.md`
5. Run training:

```bash
classification-trainer train --dataset my-dataset --training-info my-model
```

## Configuration

All configuration is stored in YAML files across six directories. Each directory contains a fully-commented `example.yaml` that documents every field — refer to those files for detailed field documentation.

### Config Architecture

**`training_info/`** is the central config hub. It references other configs by filename stem (no `.yaml` extension):

```
training_info/my-model.yaml
├── base_model: "qwen2.5-0.5b-instruct"  →  base_model_info/qwen2.5-0.5b-instruct.yaml
│   └── chat_template: "chat-ml"          →  chat_template_info/chat-ml.yaml
├── inference: "simple-classification"     →  inference_info/simple-classification.yaml
└── publishing: "standard-publish"         →  publishing_info/standard-publish.yaml  (or null to skip)
```

**`dataset_info/`** is passed separately via the `--dataset` CLI flag.

### Config Directories

| Directory | Purpose | Referenced by |
|---|---|---|
| `training_info/` | Training hyperparameters, model identity, wandb config, references to other configs | CLI `--training-info` |
| `dataset_info/` | HuggingFace dataset name, column mapping, splits, classification settings | CLI `--dataset` |
| `base_model_info/` | Pretrained model to fine-tune, chat template reference | `training_info.base_model` |
| `chat_template_info/` | Turn separators, stop strings, tokenizer settings per model family | `base_model_info.chat_template` |
| `inference_info/` | Generation parameters for evaluation (temperature, top_p, max_new_tokens) | `training_info.inference` |
| `publishing_info/` | Which formats to save/publish (LoRA, GGUF, merged), quantization levels | `training_info.publishing` |

### Naming Conventions

- All config filenames use **kebab-case** (e.g., `my-rpg-classifier.yaml`)
- CLI flags accept the filename stem without `.yaml` (e.g., `--dataset my-rpg-classifier`)

### System Prompts

System prompts are Markdown files stored in `fragments/`. The `system_prompt_name` field in `training_info` specifies the filename (with `.md` extension). The prompt is injected as the system message in the chat template during both training and evaluation.

## Recommended Workflow

### Step 1: Create configs

Copy the `example.yaml` from `dataset_info/` and `training_info/` (and other config directories as needed). Fill in your values.

### Step 2: Analyze your dataset

Determine the token length distribution of your dataset to set an appropriate `max_sequence_length`:

```bash
classification-trainer analyze-dataset --dataset my-dataset --training-info my-model
```

Update `max_sequence_length` in your training config based on the output.

### Step 3: Find the optimal batch size

Binary-search for the largest batch size that fits in GPU memory:

```bash
classification-trainer compute-batch-size --dataset my-dataset --training-info my-model
```

Update `per_device_batch_size` in your training config with the result.

### Step 4: Validate with a short training run

Do a quick test run to make sure everything works end-to-end. Set `training_length` to a small value (e.g., `0.15` epochs) and run:

```bash
classification-trainer train --dataset my-dataset --training-info my-model
```

### Step 5: Run a hyperparameter sweep

Use wandb sweeps to find optimal hyperparameters:

```bash
classification-trainer sweep --dataset my-dataset --training-info my-model --count 20
```

Review results in the wandb dashboard and update your training config with the best parameters.

### Step 6: Final training run

Set `training_length` to the full value, update hyperparameters from the sweep, enable publishing in your training config (`publishing: "my-publish-config"`), and train:

```bash
classification-trainer train --dataset my-dataset --training-info my-model
```

### Step 7: Publish to HuggingFace Hub

If publishing wasn't configured during training, or if you want to publish additional formats:

```bash
classification-trainer publish --training-info my-model --publishing-info my-publish-config
```

## CLI Reference

All commands are run via `classification-trainer` (or `python -m classification_trainer`).

### `analyze-dataset`

Analyze the token length distribution of a dataset.

```
classification-trainer analyze-dataset --dataset <name> --training-info <name> [--all-splits]
```

| Option | Description | Default |
|---|---|---|
| `--dataset` | Dataset info config name (required) | — |
| `--training-info` | Training info config name (required) | — |
| `--all-splits` | Analyze all splits instead of just training | `false` |

### `compute-batch-size`

Find the largest batch size that fits in GPU memory.

```
classification-trainer compute-batch-size --dataset <name> --training-info <name> [--stress-set-rowcount N]
```

| Option | Description | Default |
|---|---|---|
| `--dataset` | Dataset info config name (required) | — |
| `--training-info` | Training info config name (required) | — |
| `--stress-set-rowcount` | Number of longest sequences to use for stress testing | `100` |

### `train`

Run a training job.

```
classification-trainer train --dataset <name> --training-info <name> [--run-comparison-before-training / --no-run-comparison-before-training]
```

| Option | Description | Default |
|---|---|---|
| `--dataset` | Dataset info config name (required) | — |
| `--training-info` | Training info config name (required) | — |
| `--run-comparison-before-training` | Evaluate on test set before and after training to compare | `true` |

### `sweep`

Run a hyperparameter sweep using wandb.

```
classification-trainer sweep --dataset <name> --training-info <name> [--count N]
```

| Option | Description | Default |
|---|---|---|
| `--dataset` | Dataset info config name (required) | — |
| `--training-info` | Training info config name (required) | — |
| `--count` | Maximum number of sweep trials | `10` |

### `publish`

Publish saved model artifacts to HuggingFace Hub.

```
classification-trainer publish --training-info <name> --publishing-info <name>
```

| Option | Description | Default |
|---|---|---|
| `--training-info` | Training info config name (required) | — |
| `--publishing-info` | Publishing info config name (required) | — |

### `--version`

```
classification-trainer --version
```

## Output Directory Structure

```
outputs/                              # Training checkpoints and logs
└── <model_name>/
    ├── checkpoint-50/                # Checkpoints saved every N steps
    ├── checkpoint-100/
    └── <run_id>/                     # Per-trial directories (sweep mode)

outputs/                              # Saved model artifacts
└── <model_name>/
    ├── lora/                         # LoRA adapter weights
    ├── gguf/                         # GGUF quantized model files
    │   ├── <model_name>-gguf-q8_0.gguf
    │   └── <model_name>-gguf-q4_k_m.gguf
    └── merged/                       # Full merged checkpoint

exploration_reports/                  # Dataset analysis output
```

HuggingFace repos are named `<hf_username>/<model_name>-<format>` (e.g., `eriksalt/my-classifier-lora`).

## Extending: Custom Metrics

The evaluation system uses two protocols that you can implement to add new metrics and new reporting destinations.

### `MetricProtocol` — Computing a Metric

A metric computes a single scalar value from the binary classification confusion matrix. The protocol is defined in `src/classification_trainer/helpers/evaluation_helper.py`:

```python
class MetricProtocol(Protocol):
    def compute_metric(self, counts: ClassificationCounts) -> MetricResult: ...
```

`ClassificationCounts` provides `true_positives`, `false_positives`, `true_negatives`, `false_negatives`, and `total`. `MetricResult` is a `NamedTuple(metric_name: str, metric_result: Any)`.

**To add a new metric:**

1. Create a class that implements `compute_metric`:

```python
# src/classification_trainer/helpers/evaluation_helper.py

class SpecificityMetric:
    """Computes specificity: TN / (TN + FP). Returns 0.0 when denominator is 0."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        denominator = counts.true_negatives + counts.false_positives
        if denominator == 0:
            return MetricResult("specificity", 0.0)
        return MetricResult("specificity", counts.true_negatives / denominator)
```

2. Register it in `_METRIC_REGISTRY` in the same file:

```python
_METRIC_REGISTRY: dict[str, MetricProtocol] = {
    "accuracy": AccuracyMetric(),
    "precision": PrecisionMetric(),
    "recall": RecallMetric(),
    "f1": F1Metric(),
    "total_seen": TotalMetric(),
    "specificity": SpecificityMetric(),  # add here
}
```

3. Add the metric name to the `metrics` list in your `inference_info` YAML:

```yaml
metrics:
  - accuracy
  - precision
  - recall
  - f1
  - specificity
```

The metric will now be computed during evaluation and reported through all active reporters.

### `MetricsReportingProtocol` — Reporting Results

A reporter receives computed metric results and sends them somewhere. The protocol is defined in `src/classification_trainer/protocols/metric_reporting_protocol.py`:

```python
class MetricsReportingProtocol(Protocol):
    def report(self, results: Iterable[MetricResult], step: int) -> None: ...
```

Built-in reporters (in `src/classification_trainer/helpers/reporting_helper.py`):

| Reporter | Destination |
|---|---|
| `LoggerMetricsReporter` | Console via the `LoggingProtocol` |
| `WandBMetricsReporter` | Weights & Biases via `wandb.log()` |
| `CompositeMetricsReporter` | Fans out to a list of other reporters |

**To add a new reporter:**

1. Create a class that implements `report`:

```python
# src/classification_trainer/helpers/reporting_helper.py

@dataclass
class CsvMetricsReporter(MetricsReportingProtocol):
    path: Path

    def report(self, results: Iterable[MetricResult], step: int) -> None:
        result_list = list(results)
        with open(self.path, "a") as f:
            for r in result_list:
                f.write(f"{step},{r.metric_name},{r.metric_result}\n")
```

2. Add it to the reporter list in the command that should use it. In `src/classification_trainer/commands/train.py`, the reporters are assembled around line 167:

```python
reporters: list[MetricsReportingProtocol] = [LoggerMetricsReporter(logger)]
if self.training_info.wandb_config is not None:
    reporters.append(WandBMetricsReporter())
reporters.append(CsvMetricsReporter(Path("metrics.csv")))  # add here
reporter = CompositeMetricsReporter(reporters)
```

The `CompositeMetricsReporter` calls each reporter in sequence, so your new reporter will receive the same `MetricResult` values as the built-in ones.

## Development

```bash
# Run tests
cd src
pytest

# Lint
ruff check .
```
