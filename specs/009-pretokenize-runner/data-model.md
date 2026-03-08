# Data Model: Pre-tokenize Runner

## New Dataset Columns

### Training Tokenization Columns

| Column | Type | Description | When Added |
|--------|------|-------------|------------|
| `input_ids` | `list[int]` | Token IDs from tokenizing the training text column. Truncated to `max_sequence_length`. | `pretokenize=True` |
| `attention_mask` | `list[int]` | 1/0 mask matching `input_ids` length. All 1s (no padding at this stage). | `pretokenize=True` |
| `labels` | `list[int]` | Copy of `input_ids` with instruction tokens (up to and including response separator) set to `-100`. | `pretokenize=True` AND `train_on_outputs_only=True` |

### Eval Tokenization Columns

| Column | Type | Description | When Added |
|--------|------|-------------|------------|
| `eval_input_ids` | `list[int]` | Token IDs from tokenizing the eval prompt column (with prompt cleaning applied). No truncation. | `pretokenize=True` |
| `eval_attention_mask` | `list[int]` | 1/0 mask matching `eval_input_ids` length. All 1s. | `pretokenize=True` |

## Modified Entities

### TrainingRunner (dataclass)

New field:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pretokenize` | `bool` | `False` | When True, `prepare_data` adds tokenization columns to all splits. |

### DatasetInfo.get_generated_column_names()

Add to yielded names: `input_ids`, `attention_mask`, `labels`, `eval_input_ids`, `eval_attention_mask`.

## Data Flow

```
pretokenize=False (default):
  prepare_data → prep_dataset → [text columns only] → SFTTrainer tokenizes on-the-fly

pretokenize=True:
  prepare_data → prep_dataset → tokenize_training_column → apply_response_masking (if needed)
                              → tokenize_eval_column
                → [text + token columns] → SFTTrainer detects input_ids, skips tokenization
                                         → inference detects eval_input_ids, skips tokenization
```

## Tokenization Parameters

### Training Column Tokenization
- `truncation=True`
- `max_length=training_info.max_sequence_length`
- `add_special_tokens=False` (chat template already adds specials)

### Eval Column Tokenization
- No truncation
- `add_special_tokens=chat_template_info.add_special_tokens`
- Prompt text cleaned via `_clean_prompt_ending()` before tokenization

### Response Masking
- Separator: tokenized form of `chat_template_info.response_separator`
- Algorithm: find last occurrence of separator token subsequence in `input_ids`
- Mask: positions `[0, separator_end)` → `-100`; positions `[separator_end, end)` → `input_ids` values
- Missing separator: all positions → `-100`
