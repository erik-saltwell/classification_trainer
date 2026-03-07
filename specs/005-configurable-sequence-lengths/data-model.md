# Data Model: Configurable Sequence Length Analysis

**Feature**: 005-configurable-sequence-lengths | **Date**: 2026-03-07

## Modified Entities

### DatasetInfo

| Field | Change | Type | Default | Description |
|-------|--------|------|---------|-------------|
| `sequence_lengths` | ADDED | `list[int]` | `[1024, 1536, 2048]` | Ordered list of positive integers specifying token-count thresholds for the coverage report in the analyze-dataset command. |

**Validation rules**:
- List must not be empty.
- All values must be positive integers (> 0).
- Duplicates are removed while preserving order.

**Behavior**: The analyze-dataset command iterates `dataset_info.sequence_lengths` instead of hardcoded values. When the field is omitted from YAML, the default `[1024, 1536, 2048]` applies — identical to current behavior.
