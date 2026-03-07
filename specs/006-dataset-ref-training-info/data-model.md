# Data Model: Dataset Reference in Training Config

**Feature**: 006-dataset-ref-training-info | **Date**: 2026-03-07

## Modified Entities

### TrainingInfo

| Field | Change | Type | Required | Description |
|-------|--------|------|----------|-------------|
| `dataset` | ADDED | `str` | Yes | Filename stem (no .yaml extension) of the dataset config to use. Must match a file in `dataset_info/`. |

**New property**:
- `dataset_info -> DatasetInfo`: Loads and returns the `DatasetInfo` from `dataset_info/<dataset>.yaml`. Follows the same lazy-loading pattern as `base_model_info`, `inference_info`, and `publishing_info`.

**Validation**: Pydantic requires the field at parse time (no default). If the referenced file doesn't exist, the property raises `FileNotFoundError` with a clear message.

### Command Classes (all modified)

| Class | Change | Notes |
|-------|--------|-------|
| `AnalyzeDatasetCommand` | REMOVE `dataset_info` param | Use `self.training_info.dataset_info` |
| `TrainCommand` | REMOVE `dataset_info` param | Use `self.training_info.dataset_info` |
| `SweepCommand` | REMOVE `dataset_info` param | Use `self.training_info.dataset_info` |
| `ComputeBatchSizeCommand` | REMOVE `dataset_info` param | Use `self.training_info.dataset_info` |

### CLI (console/main.py)

| Command | Change |
|---------|--------|
| `analyze-dataset` | Remove `--dataset` option |
| `train` | Remove `--dataset` option |
| `sweep` | Remove `--dataset` option |
| `compute-batch-size` | Remove `--dataset` option |

All four commands resolve dataset via `tr_info.dataset_info` after loading training info.

## Relationships

```text
TrainingInfo
├── dataset: str                → dataset_info property → DatasetInfo
├── base_model: str             → base_model_info property → BaseModelInfo
├── inference: str              → inference_info property → InferenceInfo
└── publishing: str | None      → publishing_info property → PublishingInfo | None
```
