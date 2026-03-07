# Contract: CLI Commands (after dataset reference migration)

**Feature**: 006-dataset-ref-training-info | **Date**: 2026-03-07

## Changed Commands

All four commands lose the `--dataset` argument. The dataset is resolved from the training config's `dataset` field.

### analyze-dataset

**Before**:
```
analyze-dataset --dataset <name> --training-info <name> [--all-splits]
```

**After**:
```
analyze-dataset --training-info <name> [--all-splits]
```

### train

**Before**:
```
train --dataset <name> --training-info <name> [--run-comparison-before-training]
```

**After**:
```
train --training-info <name> [--run-comparison-before-training]
```

### sweep

**Before**:
```
sweep --dataset <name> --training-info <name> [--count N]
```

**After**:
```
sweep --training-info <name> [--count N]
```

### compute-batch-size

**Before**:
```
compute-batch-size --dataset <name> --training-info <name> [--stress-set-rowcount N]
```

**After**:
```
compute-batch-size --training-info <name> [--stress-set-rowcount N]
```

## Training Config YAML Addition

```yaml
# References to reusable config files (filename stem, no .yaml extension)
base_model: "qwen2.5-0.5b-instruct"
dataset: "imdb"                    # NEW — dataset_info/<name>.yaml
inference: "simple-classification"
publishing: null
```
