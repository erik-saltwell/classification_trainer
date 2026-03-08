# Implementation Plan: Pre-tokenize Runner

**Branch**: `009-pretokenize-runner` | **Date**: 2026-03-08 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/009-pretokenize-runner/spec.md`

## Summary

Add a `pretokenize` boolean parameter (default `False`) to `TrainingRunner`. When enabled, `prepare_data()` tokenizes the training and eval text columns into `input_ids`, `attention_mask`, `labels`, `eval_input_ids`, and `eval_attention_mask` columns. Downstream, `create_trainer()` auto-detects `input_ids` and configures SFTTrainer to skip re-tokenization. The inference path auto-detects `eval_input_ids` and generates directly from tokens. No existing callers are changed.

**Key research finding**: SFTTrainer auto-detects `input_ids` in the dataset and skips tokenization entirely. TRL's own data collator preserves pre-computed `labels` (unlike the transformers library version). When labels are pre-masked, `train_on_responses_only()` must be skipped as it would overwrite them. See [research.md](research.md) for full details and sources.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: Unsloth `FastLanguageModel`, TRL `SFTTrainer`, HuggingFace `datasets` + `transformers`, Pydantic v2
**Storage**: HuggingFace Dataset (in-memory, Arrow-backed)
**Testing**: pytest (unit tests in `tests/unit/`)
**Target Platform**: Linux with NVIDIA GPU
**Project Type**: CLI tool
**Performance Goals**: Eliminate redundant tokenization when `pretokenize=True`; no performance regression when `pretokenize=False`
**Constraints**: GPU memory budget unchanged; dataset memory footprint increases slightly with additional token columns
**Scale/Scope**: Affects TrainingRunner only; no new CLI commands or config files

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Configuration-First | PASS | No new YAML files or hardcoded parameters. `pretokenize` is a code parameter on TrainingRunner, not a training hyperparameter. No new directories needed. |
| II. Protocol-Based Interfaces | PASS | No new protocols needed. Existing `LoggingProtocol` usage unchanged. |
| III. Separation of Concerns | PASS | New tokenization functions go in `helpers/dataset_helper.py` (domain logic). Detection logic in `helpers/training_helper.py` and `helpers/inference_helper.py` (domain logic). TrainingRunner parameter addition is orchestration-level. |
| IV. Observability | PASS | No new metrics or logging requirements. Existing metrics reporting unchanged. |
| V. Simplicity & Scope | PASS | Feature is within classification fine-tuning scope. New functions are minimal. No new abstractions — just conditional branches on column presence. |

**Post-Phase 1 re-check**: All gates still pass. No new directories, protocols, or abstractions introduced.

## Project Structure

### Documentation (this feature)

```text
specs/009-pretokenize-runner/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # SFTTrainer pre-tokenization research
├── data-model.md        # New columns and entity changes
├── quickstart.md        # Usage guide
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Task list (created by /speckit.tasks)
```

### Source Code (repository root)

```text
src/classification_trainer/
├── commands/
│   └── training_runner.py      # MODIFY: add pretokenize parameter
├── configuration/
│   ├── dataset_info.py          # MODIFY: add token column names to generated columns
│   └── training_info.py         # MODIFY: add pre_tokenized param to create_sft_config
├── helpers/
│   ├── dataset_helper.py        # MODIFY: add tokenize_training_column, apply_response_masking,
│   │                            #          tokenize_eval_column; update prep_dataset
│   ├── training_helper.py       # MODIFY: auto-detect pre-tokenized in create_trainer
│   └── inference_helper.py      # MODIFY: add generate_label_text_from_tokens;
│                                #          update add_inferred_column

tests/unit/
├── test_dataset_helper.py       # MODIFY: add tests for new tokenization functions
├── test_dataset_info.py         # MODIFY: add test for new generated column names
└── test_training_info.py        # MODIFY: add test for pre_tokenized param
```

**Structure Decision**: Single project, existing directory layout. All changes are modifications to existing files. No new files or directories in source code.

## Files to Modify

### 1. `src/classification_trainer/helpers/dataset_helper.py`

**New functions** (add after `add_eval_column`):

- `tokenize_training_column(dataset_info, dataset, tokenizer, max_seq_len) -> Dataset`
  - Tokenizes training text column via `dataset.map`, adds `input_ids` and `attention_mask`
  - Uses `truncation=True, max_length=max_seq_len, add_special_tokens=False`

- `_find_last_subsequence(sequence, subsequence) -> int | None`
  - Helper: finds last occurrence of a token subsequence in a token list
  - Returns start index or None

- `apply_response_masking(dataset, tokenizer, chat_template_info) -> Dataset`
  - Tokenizes the response separator string to get separator token IDs
  - For each row: finds last occurrence of separator tokens in `input_ids`
  - Sets `labels[0:sep_end] = -100`, keeps `labels[sep_end:] = input_ids[sep_end:]`
  - If separator not found: all labels = -100

- `_clean_prompt_ending(prompt_text, template) -> str`
  - Duplicated from `inference_helper.py` to avoid importing heavy unsloth/torch deps
  - Ensures prompt ends with newline after response separator when `assistant_newline=True`

- `tokenize_eval_column(dataset_info, dataset, tokenizer, chat_template_info) -> Dataset`
  - Applies `_clean_prompt_ending` then tokenizes eval prompts
  - Adds `eval_input_ids` and `eval_attention_mask` columns
  - Uses `add_special_tokens=chat_template_info.add_special_tokens`, no truncation

**Modified function**: `prep_dataset()`
- Add `pretokenize: bool = False` parameter
- When True, after existing text formatting, call:
  1. `tokenize_training_column(...)`
  2. `apply_response_masking(...)` (if `training_info.train_on_outputs_only`)
  3. `tokenize_eval_column(...)`

**Modified function**: `prepare_split_data()`
- Add `pretokenize: bool = False` parameter, pass through to `prep_dataset()`

### 2. `src/classification_trainer/configuration/training_info.py`

**Modified function**: `create_sft_config()`
- Add `pre_tokenized: bool = False` parameter
- When True: set `dataset_text_field=None` instead of `dataset_info.training_column_name`

### 3. `src/classification_trainer/configuration/dataset_info.py`

**Modified method**: `get_generated_column_names()`
- Add yields for: `input_ids`, `attention_mask`, `labels`, `eval_input_ids`, `eval_attention_mask`

### 4. `src/classification_trainer/helpers/training_helper.py`

**Modified function**: `create_trainer()`
- Auto-detect: `pre_tokenized = "input_ids" in train_dataset.column_names`
- Pass `pre_tokenized` to `create_sft_config()`
- Skip `train_on_responses_only()` when `pre_tokenized` is True (labels already masked)

### 5. `src/classification_trainer/helpers/inference_helper.py`

**New function**: `generate_label_text_from_tokens(model, tokenizer, input_ids, attention_mask, inference_info, template) -> str`
- Takes `input_ids` and `attention_mask` as `list[int]`
- Converts to tensors, calls `model.generate()`, decodes and trims

**Modified function**: `add_inferred_column()`
- Detect `eval_input_ids` in dataset columns
- When present: use `generate_label_text_from_tokens()` per row instead of `generate_label_texts()` on text
- When absent: unchanged behavior (text-based tokenization)

### 6. `src/classification_trainer/commands/training_runner.py`

**Modified class**: `TrainingRunner`
- Add `pretokenize: bool = False` field
- Pass `pretokenize` through `prepare_data()` → `prepare_split_data()` → `prep_dataset()`

### 7. Test files

- `tests/unit/test_dataset_helper.py`: Tests for `tokenize_training_column`, `apply_response_masking`, `tokenize_eval_column`, `_find_last_subsequence`
- `tests/unit/test_dataset_info.py`: Test that new column names appear in `get_generated_column_names()`
- `tests/unit/test_training_info.py`: Test `create_sft_config(pre_tokenized=True)` produces `dataset_text_field=None`

## Key Design Decisions

### D1: Column detection over explicit flags
The trainer and inference code detect pre-tokenized data by checking for `input_ids` and `eval_input_ids` columns respectively. This is more robust than threading a boolean flag through every function call and aligns with how SFTTrainer itself detects pre-tokenized data.

### D2: Duplicating `_clean_prompt_ending`
Rather than importing from `inference_helper.py` (which pulls in torch and unsloth), the small prompt-cleaning function is duplicated in `dataset_helper.py`. This keeps the dataset helper free of heavy GPU dependencies.

### D3: Last occurrence of separator
The response masking algorithm finds the **last** occurrence of the separator token subsequence. For single-turn classification this is equivalent to first occurrence, but is defensive against edge cases.

### D4: `pretokenize` as TrainingRunner parameter (not config)
The `pretokenize` flag is a TrainingRunner code parameter, not a YAML config setting. This is correct per Constitution Principle I: it's a runtime optimization choice, not a training hyperparameter that affects reproducibility.

## Known Limitations

- `pretokenize=True` with `packing=True` is untested and may not work correctly (packing may overwrite labels)
- Sweep and compute-batch-size commands do not use TrainingRunner and are not affected by this feature; they will be updated in a future feature
- Slight increase in dataset memory usage due to additional token columns alongside text columns

## Complexity Tracking

No constitution violations. No complexity tracking entries needed.
