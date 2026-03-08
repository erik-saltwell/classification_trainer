# Quickstart: Pre-tokenize Runner

## Usage

```python
from classification_trainer.commands.training_runner import TrainingRunner

# Default behavior (unchanged):
runner = TrainingRunner(training_info, dataset_info)

# With pre-tokenization enabled:
runner = TrainingRunner(training_info, dataset_info, pretokenize=True)
```

When `pretokenize=True`:
- `prepare_data()` adds token columns (`input_ids`, `attention_mask`, `labels`, `eval_input_ids`, `eval_attention_mask`) to all dataset splits
- `train_model()` auto-detects `input_ids` and skips re-tokenization
- `evaluate_model()` auto-detects `eval_input_ids` and skips re-tokenization of prompts

No changes to existing callers are required. The default is `False`.

## Verification

```bash
cd src && pytest
```

All existing tests should pass unchanged since `pretokenize` defaults to `False`.
