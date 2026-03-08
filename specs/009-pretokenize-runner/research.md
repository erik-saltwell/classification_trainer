# Research: Pre-tokenize Runner

## R1: How SFTTrainer Handles Pre-Tokenized Datasets

### Decision
When the dataset contains an `input_ids` column, SFTTrainer auto-detects it and skips all tokenization. The `dataset_text_field` parameter becomes irrelevant. Setting it to `None` (or any value) has no effect when `input_ids` is present.

### Mechanism
SFTTrainer's `_prepare_dataset` method checks:
```python
column_names = get_dataset_column_names(dataset)
is_processed = "input_ids" in column_names
```
When `is_processed` is `True`, the dataset is used as-is with no formatting or tokenization applied. If a formatting function is also provided, a warning is emitted and the formatter is ignored.

### Rationale
- This is the simplest approach — no special configuration needed. Just ensure `input_ids` is in the dataset.
- Setting `dataset_text_field=None` is the cleanest way to signal intent, but not strictly required.

### Alternatives Considered
- **Explicit `skip_tokenization` flag in SFTConfig**: Not supported by TRL.
- **Custom data collator**: Unnecessary since TRL's built-in behavior handles this.

### Version Compatibility
- Pre-tokenized support has existed since TRL ~v0.9.6.
- **TRL v0.15.0 had a regression** (issue #2861) that broke pre-tokenized support with `KeyError: 'text'`. Fixed by PR #2863 (merged Feb 2025).
- Any TRL version post-fix should work correctly.

### Sources
- [TRL issue #2861](https://github.com/huggingface/trl/issues/2861)
- [TRL issue #991](https://github.com/huggingface/trl/issues/991)
- [TRL issue #794](https://github.com/huggingface/trl/issues/794)
- [TRL SFTTrainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py)

---

## R2: TRL's Data Collator Preserves Pre-Computed Labels

### Decision
TRL defines its own `DataCollatorForLanguageModeling` (distinct from the transformers library version). It **preserves existing `labels`** from the dataset:

```python
if "labels" in examples[0]:
    labels = [torch.tensor(example["labels"]) for example in examples]
else:
    labels = [torch.tensor(example["input_ids"]) for example in examples]
```

### Rationale
- This is critical: the transformers library's `DataCollatorForLanguageModeling` has a known bug where it always clones `input_ids` as labels, overwriting user-provided `-100` values ([transformers issue #26357](https://github.com/huggingface/transformers/issues/26357)).
- TRL's version does **not** have this bug.
- Pre-computed `labels` with `-100` masking will be used as-is.

### Padding Behavior
- `input_ids`: padded to batch max length with `pad_token_id`
- `labels`: padded with `-100` (PyTorch CrossEntropyLoss ignore index)
- `attention_mask`: auto-generated as all-ones per sequence length, padded with `0`

Variable-length Python lists of `input_ids` and `labels` stored in HuggingFace Datasets are correctly handled — they are converted to tensors and dynamically padded per-batch.

### Caveat: completion_mask
If `completion_only_loss=True` and `completion_mask` is present, the collator **overwrites** labels for non-completion tokens with `-100`. This project does not use `completion_only_loss`, so this is not a concern.

---

## R3: Interaction with `train_on_responses_only()`

### Decision
When labels are pre-computed with `-100` masking, **skip `train_on_responses_only()` entirely**. The function wraps the data collator and would overwrite the pre-computed labels.

### How `train_on_responses_only()` Works
1. Wraps/modifies the trainer's data collator
2. At collation time, decodes `input_ids` back to text
3. Searches for `instruction_part` and `response_part` string delimiters in the decoded text
4. Sets labels to `-100` for all tokens outside response regions

### Why Skip It
- It **overwrites** any existing `labels` column values ([unsloth issue #1017](https://github.com/unslothai/unsloth/issues/1017))
- When labels are already correctly masked, applying the function again is redundant at best and destructive at worst
- The pre-computed labels are more reliable because they operate on exact token boundaries rather than string matching on decoded text

### Implementation Approach
In `create_trainer()`, detect pre-tokenized data by checking `"input_ids" in train_dataset.column_names`. When detected:
1. Pass `pre_tokenized=True` to `create_sft_config()` so `dataset_text_field` is set to `None`
2. Skip the `train_on_responses_only()` call (labels are already masked)

### Sources
- [Unsloth issue #1017](https://github.com/unslothai/unsloth/issues/1017)
- [Unsloth discussion #2828](https://github.com/unslothai/unsloth/discussions/2828)

---

## R4: Pre-Tokenized Eval Data for Inference

### Decision
For evaluation/inference, pre-tokenize the eval prompt column into `eval_input_ids` and `eval_attention_mask`. At inference time, detect these columns and pass the token tensors directly to `model.generate()` instead of re-tokenizing the text.

### Current Inference Flow
1. `add_inferred_column()` reads the text from `evaluation_instructions_column_name`
2. `generate_label_text()` tokenizes each prompt via `tokenizer(prompt_text, ...)`
3. `model.generate()` produces output
4. Generated tokens are decoded and trimmed

### Pre-Tokenized Inference Flow
1. `add_inferred_column()` detects `eval_input_ids` in the dataset
2. A new function `generate_label_text_from_tokens()` takes `input_ids` and `attention_mask` as lists, converts to tensors, and calls `model.generate()`
3. Generated tokens are decoded and trimmed (same as current)

### Important: Prompt Cleaning Before Tokenization
The current text path applies `_clean_prompt_ending()` (which ensures a newline after the response separator when `assistant_newline=True`) before tokenizing. The pre-tokenization path must apply this same cleaning to the eval prompt text **before** tokenizing it into `eval_input_ids`. This ensures identical generation behavior.

### Alternatives Considered
- **Keep eval text-based**: Simpler but misses the performance benefit for evaluation.
- **Batch tokenization at inference time**: Would require accumulating prompts, which changes the existing map-based architecture.

---

## R5: Response Masking Algorithm

### Decision
Implement manual label masking by finding the response separator token subsequence in `input_ids`, rather than relying on `train_on_responses_only()`.

### Algorithm
1. Tokenize the response separator string (e.g., `<|im_start|>assistant\n`) to get its token ID sequence
2. For each row's `input_ids`, find the **last** occurrence of this subsequence
3. Set `labels[0:separator_end]` to `-100` (mask instruction tokens)
4. Set `labels[separator_end:]` to `input_ids[separator_end:]` (keep response tokens for training)
5. If the separator is not found, set all labels to `-100` (matches existing behavior where missing separators produce NaN loss — caught by upstream validation)

### Why Last Occurrence
Classification conversations are single-turn (system + user + assistant). Using the last occurrence is defensive against edge cases where separator-like strings appear earlier in the text.

### Alternatives Considered
- **Using `train_on_responses_only(return_function=True)`**: This option was explored but the function returns a wrapped collator, not a mapping function. It cannot be used to pre-compute labels on a dataset.
- **Using both instruction and response separators**: More complex but matches how `train_on_responses_only()` works. For single-turn classification, finding the response separator alone is sufficient and simpler.
