# Research: Configurable Sequence Length Analysis

**Feature**: 005-configurable-sequence-lengths | **Date**: 2026-03-07

## R1: Field Placement

**Decision**: Add `sequence_lengths` as an optional field on `DatasetInfo` with default `[1024, 1536, 2048]`.

**Rationale**: The sequence lengths are a property of how the dataset is analyzed, which is dataset-specific. Placing it on `DatasetInfo` follows the existing pattern — `DatasetInfo` already contains analysis-related fields like `search_length` and `search_from_end`. The default matches the current hardcoded values for backward compatibility.

**Alternatives considered**:
- CLI argument: Rejected — the user description explicitly says "specified in the datasetinfo yaml", and CLI args are more ephemeral than config files.
- TrainingInfo: Rejected — the analyze-dataset command operates on dataset properties, not training hyperparameters.

## R2: Validation Strategy

**Decision**: Use Pydantic `field_validator` on `DatasetInfo` to enforce: non-empty list, all positive integers. Deduplication via a validator that removes duplicates while preserving order.

**Rationale**: Consistent with existing validation patterns in `DatasetInfo` (e.g., `validate_max_rowcount`, `validate_column_name`). Pydantic validation fires at config load time, catching errors before any analysis runs.

## R3: Type Declaration

**Decision**: `sequence_lengths: list[int] = [1024, 1536, 2048]`

**Rationale**: Pydantic v2 will reject float values passed to `list[int]` at parse time. Combined with the positive-integer validator, this covers all edge cases from the spec (non-integers, non-positive, empty list).
