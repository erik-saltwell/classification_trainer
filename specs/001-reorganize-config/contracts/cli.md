# CLI Contract: Configuration File Reorganization

**Branch**: `001-reorganize-config` | **Date**: 2026-03-05

## Summary of Changes

Three arguments (`--base-model`, `--inference-info`, `--publishing-info`) are removed from the commands where they currently appear. These configs are now declared inside the `training_info` YAML and resolved automatically. The `publish` command retains `--publishing-info` as an explicit override.

---

## Command Contracts

### `train` (changed)

**Before**:
```
classification-trainer train \
  --dataset <name> \
  --base-model <name> \
  --training-info <name> \
  --inference-info <name> \
  [--publishing-info <name>]
```

**After**:
```
classification-trainer train \
  --dataset <name> \
  --training-info <name>
```

`base_model`, `inference`, and `publishing` are resolved from the training config YAML.

---

### `sweep` (changed)

**Before**:
```
classification-trainer sweep \
  --dataset <name> \
  --base-model <name> \
  --training-info <name> \
  --inference-info <name> \
  [--count <int>]
```

**After**:
```
classification-trainer sweep \
  --dataset <name> \
  --training-info <name> \
  [--count <int>]
```

---

### `analyze-dataset` (changed)

**Before**:
```
classification-trainer analyze-dataset \
  --dataset <name> \
  --base-model <name> \
  --training-info <name> \
  [--all-splits]
```

**After**:
```
classification-trainer analyze-dataset \
  --dataset <name> \
  --training-info <name> \
  [--all-splits]
```

---

### `compute-batch-size` (changed)

**Before**:
```
classification-trainer compute-batch-size \
  --dataset <name> \
  --base-model <name> \
  --training-info <name> \
  [--stress-set-rowcount <int>]
```

**After**:
```
classification-trainer compute-batch-size \
  --dataset <name> \
  --training-info <name> \
  [--stress-set-rowcount <int>]
```

---

### `publish` (unchanged)

```
classification-trainer publish \
  --training-info <name> \
  --publishing-info <name>
```

Rationale: `publish` runs independently of a training session; the user may legitimately want to re-publish existing model artifacts with different format settings without modifying their training config.

---

## Error Contract

When a config name referenced inside `training_info` does not resolve to an existing YAML file, the system MUST exit with a non-zero code and print:

```
Error: <config-category> config not found: '<name>'
Expected location: <directory>/<name>.yaml
```

This matches the existing error format produced by `load_config_or_exit()` in `console_validation.py`.
