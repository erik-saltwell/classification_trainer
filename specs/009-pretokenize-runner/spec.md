# Feature Specification: Pre-tokenize Runner

**Feature Branch**: `009-pretokenize-runner`
**Created**: 2026-03-07
**Status**: Draft
**Input**: User description: "Add a boolean `pretokenize` parameter to TrainingRunner. When set, it generates tokenized versions of the training and eval columns during data preparation. Training and evaluation then use the pre-tokenized columns instead of re-tokenizing. Default is False. Do not change any existing callers."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Default Behavior Unchanged (Priority: P1)

A developer using TrainingRunner without specifying the `pretokenize` parameter experiences identical behavior to today. The training and evaluation pipeline tokenizes data on-the-fly as it currently does. No existing callers need to be updated.

**Why this priority**: Backward compatibility is the most critical requirement. Existing training and evaluation workflows must not break.

**Independent Test**: Run the existing train command end-to-end and verify output is identical to the current behavior.

**Acceptance Scenarios**:

1. **Given** TrainingRunner is created without the `pretokenize` parameter, **When** `prepare_data`, `train_model`, and `evaluate_model` are called in sequence, **Then** the pipeline behaves identically to the current implementation (on-the-fly tokenization).
2. **Given** TrainingRunner is created with `pretokenize=False`, **When** the pipeline runs, **Then** the prepared dataset does not contain `input_ids`, `attention_mask`, `labels`, `eval_input_ids`, or `eval_attention_mask` columns.

---

### User Story 2 - Pre-tokenize Training Data (Priority: P1)

A developer creates a TrainingRunner with `pretokenize=True`. During `prepare_data`, the training text column is tokenized into `input_ids` and `attention_mask` columns. If response masking is enabled (`train_on_outputs_only=True`), a `labels` column is also generated with instruction tokens masked to -100. When `train_model` is called, the trainer uses these pre-tokenized columns directly and does not re-tokenize.

**Why this priority**: This is the core value of the feature -- avoiding redundant tokenization during training.

**Independent Test**: Call `prepare_data` with `pretokenize=True`, inspect the dataset columns, then call `train_model` and verify the trainer skips tokenization.

**Acceptance Scenarios**:

1. **Given** TrainingRunner with `pretokenize=True` and `train_on_outputs_only=False`, **When** `prepare_data` completes, **Then** the training dataset contains `input_ids` and `attention_mask` columns derived from the training text column.
2. **Given** TrainingRunner with `pretokenize=True` and `train_on_outputs_only=True`, **When** `prepare_data` completes, **Then** the training dataset additionally contains a `labels` column where all tokens before and including the response separator are set to -100.
3. **Given** a pre-tokenized training dataset with `input_ids` present, **When** `train_model` is called, **Then** the trainer does not re-tokenize the training text column and uses the existing `input_ids` directly.
4. **Given** a pre-tokenized training dataset with `labels` present, **When** `train_model` is called, **Then** the trainer does not re-apply response masking via the data collator wrapper.

---

### User Story 3 - Pre-tokenize Eval Data (Priority: P2)

A developer creates a TrainingRunner with `pretokenize=True`. During `prepare_data`, the evaluation prompt column is tokenized into `eval_input_ids` and `eval_attention_mask` columns. When `evaluate_model` is called, inference uses these pre-tokenized columns directly instead of re-tokenizing each prompt.

**Why this priority**: Evaluation inference also re-tokenizes every call, so pre-tokenization provides a secondary performance benefit.

**Independent Test**: Call `prepare_data` with `pretokenize=True`, then call `evaluate_model` and verify inference uses the pre-tokenized eval columns.

**Acceptance Scenarios**:

1. **Given** TrainingRunner with `pretokenize=True`, **When** `prepare_data` completes, **Then** the test dataset contains `eval_input_ids` and `eval_attention_mask` columns derived from the evaluation prompt column.
2. **Given** a pre-tokenized test dataset with `eval_input_ids` present, **When** `evaluate_model` is called, **Then** inference uses the pre-tokenized token IDs for generation instead of re-tokenizing the prompt text.
3. **Given** a pre-tokenized test dataset, **When** `evaluate_model` produces predictions, **Then** the predictions are identical to what the text-based path would produce.

---

### User Story 4 - Cleanup of Generated Columns (Priority: P3)

When generated columns are removed from a dataset (e.g., via `remove_generated_columns`), the pre-tokenization columns (`input_ids`, `attention_mask`, `labels`, `eval_input_ids`, `eval_attention_mask`) are also cleaned up.

**Why this priority**: Prevents column pollution when datasets are reused or re-prepared.

**Independent Test**: Add pre-tokenized columns to a dataset, call `remove_generated_columns`, and verify all tokenization columns are removed.

**Acceptance Scenarios**:

1. **Given** a dataset with pre-tokenization columns, **When** `remove_generated_columns` is called, **Then** `input_ids`, `attention_mask`, `labels`, `eval_input_ids`, and `eval_attention_mask` are removed.

---

### Edge Cases

- What happens when `pretokenize=True` but the training text column has not yet been generated? The system should raise a clear error during data preparation (existing validation already handles this).
- What happens when the dataset already contains an `input_ids` column from a prior run? The tokenization step should overwrite it (standard `dataset.map` behavior).
- What happens when `train_on_outputs_only=True` but the response separator is not found in a row's token IDs? The row's labels should be set entirely to -100 (matching the current behavior where missing separators cause all-masked loss, which is caught by existing validation before tokenization).
- What happens when `pretokenize=True` with `packing=True`? Pre-tokenized `labels` may be overwritten by the packing logic. This combination is not currently used and is out of scope; document it as a known limitation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: TrainingRunner MUST accept a `pretokenize` boolean parameter that defaults to `False`.
- **FR-002**: When `pretokenize=True`, `prepare_data` MUST add `input_ids` and `attention_mask` columns to all dataset splits by tokenizing the training text column with `truncation=True`, `max_length=max_sequence_length`, and `add_special_tokens=False`.
- **FR-003**: When `pretokenize=True` and `train_on_outputs_only=True`, `prepare_data` MUST add a `labels` column where all tokens up to and including the response separator are set to -100.
- **FR-004**: When `pretokenize=True`, `prepare_data` MUST add `eval_input_ids` and `eval_attention_mask` columns to all dataset splits by tokenizing the evaluation prompt column.
- **FR-005**: When training data contains `input_ids`, the trainer configuration MUST omit the text field directive so the trainer uses the pre-tokenized columns directly instead of re-tokenizing.
- **FR-006**: When training data contains pre-computed `labels`, the trainer MUST skip applying its own response masking via the data collator wrapper.
- **FR-007**: When eval data contains `eval_input_ids`, the inference path MUST use the pre-tokenized token IDs for generation instead of re-tokenizing the prompt text.
- **FR-008**: The generated column names list MUST include `input_ids`, `attention_mask`, `labels`, `eval_input_ids`, and `eval_attention_mask` so they are cleaned up by `remove_generated_columns`.
- **FR-009**: When `pretokenize=False` (the default), all behavior MUST remain identical to the current implementation. No pre-tokenization columns are added.
- **FR-010**: No existing callers of TrainingRunner (train command, analyze-dataset command) MUST be modified. The new parameter is additive only.

### Key Entities

- **Pre-tokenized Training Columns**: `input_ids` (token IDs), `attention_mask` (1/0 mask), `labels` (token IDs with -100 masking for instruction tokens). Stored as variable-length integer lists in the dataset.
- **Pre-tokenized Eval Columns**: `eval_input_ids` (token IDs for evaluation prompts), `eval_attention_mask` (1/0 mask). Stored as variable-length integer lists.
- **Response Separator Token Sequence**: The tokenized form of the chat template's response separator string. Used to determine the boundary between instruction tokens (masked to -100) and response tokens (kept for training).

## Assumptions

- The chat template already adds all necessary special tokens, so tokenization uses `add_special_tokens=False` for training columns.
- Eval prompt tokenization respects the `add_special_tokens` setting from the chat template configuration.
- Eval prompt text cleaning (ensuring correct newline after response separator) is applied before tokenization.
- Variable-length `input_ids` stored as Python lists in the dataset are handled correctly by the data collator during training (padding to batch max length).
- The combination of `pretokenize=True` with `packing=True` is out of scope and may not work correctly. This is a known limitation.
- The sweep command and compute-batch-size command are not changed in this feature. They will be updated in a future feature that builds on this foundation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All existing tests pass without modification when `pretokenize` defaults to `False`.
- **SC-002**: When `pretokenize=True`, the prepared dataset contains the expected tokenization columns (`input_ids`, `attention_mask`, and conditionally `labels`, `eval_input_ids`, `eval_attention_mask`).
- **SC-003**: When `pretokenize=True`, training produces the same loss trajectory (within numerical tolerance) as the default path over an identical dataset and configuration.
- **SC-004**: When `pretokenize=True`, evaluation produces identical classification results as the default path on the same dataset.
- **SC-005**: No existing callers of TrainingRunner require changes -- the parameter is purely additive with a safe default.
