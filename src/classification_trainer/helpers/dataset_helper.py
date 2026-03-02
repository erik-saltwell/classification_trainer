import random as _random
from pathlib import Path
from typing import Any, cast

from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.utils.logging import set_verbosity_error
from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import ChatTemplateInfo, DatasetInfo, TrainingInfo
from classification_trainer.protocols.logging_protocol import LoggingProtocol

from .token_length_helper import compute_tokens

set_verbosity_error()


def take(dataset: Dataset, count: int) -> Dataset:
    """Return a new Dataset with at most *count* records from the start."""
    return dataset.select(range(min(count, len(dataset))))


def load_dataset_from_hf(dataset_info: DatasetInfo) -> DatasetDict:
    dataset = cast(DatasetDict, load_dataset(dataset_info.huggingface_name))
    if dataset_info.max_rowcount != -1:
        dataset = DatasetDict({split: take(ds, dataset_info.max_rowcount) for split, ds in dataset.items()})
    return dataset


def load_dataset_from_disk(path: Path) -> DatasetDict:
    return cast(DatasetDict, load_from_disk(str(path)))


def save_dataset_to_disk(dataset: DatasetDict, path: Path) -> None:
    dataset.save_to_disk(str(path))


def split_dataset(
    dataset_info: DatasetInfo,
    dataset: Dataset,
    validation_percent_of_total: float,
    test_percent_of_total: float,
    seed: int,
) -> tuple[DatasetDict, DatasetInfo]:
    new_dataset_info = dataset_info.add_split()  # guarantees all three split names are set

    if test_percent_of_total + validation_percent_of_total >= 1.0:
        raise ValueError("The sum of test_percent_of_total and validation_percent_of_total must be less than 1.0")

    non_train_percent_of_total = test_percent_of_total + validation_percent_of_total
    if non_train_percent_of_total == 0.0:
        raise ValueError(
            "At least one of test_percent_of_total or validation_percent_of_total must be greater than 0.0"
        )

    validation_percent_of_non_train = validation_percent_of_total / non_train_percent_of_total

    train_nontrain_sets = dataset.train_test_split(
        test_size=non_train_percent_of_total,
        stratify_by_column=new_dataset_info.label_column_name,
        seed=seed,
    )

    # Split non-train into validation and test. test_size is the validation fraction of non-train,
    # so train_test_split's "test" key becomes the validation set and "train" key becomes the test set.
    non_train_resplit = train_nontrain_sets["test"].train_test_split(
        test_size=validation_percent_of_non_train,
        stratify_by_column=new_dataset_info.label_column_name,
        seed=seed,
    )
    validation_set = non_train_resplit["test"]
    test_set = non_train_resplit["train"]

    # These are guaranteed by add_split(); asserts narrow the types for static analysis.
    assert new_dataset_info.validation_split_name is not None
    assert new_dataset_info.test_split_name is not None

    splits = DatasetDict(
        {
            new_dataset_info.training_split_name: train_nontrain_sets["train"],
            new_dataset_info.validation_split_name: validation_set,
            new_dataset_info.test_split_name: test_set,
        }
    )
    return splits, new_dataset_info


def make_stress_split(
    dataset_info: DatasetInfo,
    dataset: Dataset,
    number_of_rows: int,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Return the *number_of_rows* rows with the longest tokenized training text.

    Useful for stress-testing memory and throughput with worst-case sequence lengths.
    """
    token_counts = [compute_tokens(text, tokenizer) for text in dataset[dataset_info.training_column_name]]
    dataset = dataset.add_column("_token_count", token_counts)  # pyright: ignore
    dataset = dataset.sort("_token_count", reverse=True)
    dataset = dataset.select(range(min(number_of_rows, len(dataset))))
    dataset = dataset.remove_columns("_token_count")
    return dataset


def filter_by_sequence_length(
    dataset_info: DatasetInfo,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_sequence_length: int,
) -> Dataset:
    """Return a new Dataset containing only rows whose training column
    tokenizes to at most *max_sequence_length* tokens.

    Rows exceeding the limit are dropped entirely. Uses
    dataset_info.training_column_name as the text source, so this
    function must be called after add_training_column().
    """
    return dataset.filter(
        lambda row: compute_tokens(row[dataset_info.training_column_name], tokenizer) <= max_sequence_length
    )


def union_datasets(*datasets: Dataset) -> Dataset:
    """Concatenate any number of Datasets after validating they share the same columns.

    Raises:
        ValueError: If fewer than two datasets are provided or column names differ.
    """
    if len(datasets) < 2:
        raise ValueError(f"union_datasets requires at least 2 datasets, got {len(datasets)}")

    reference_cols = set(datasets[0].column_names)
    for i, ds in enumerate(datasets[1:], start=2):
        ds_cols = set(ds.column_names)
        if ds_cols != reference_cols:
            only_ref = reference_cols - ds_cols
            only_ds = ds_cols - reference_cols
            parts: list[str] = []
            if only_ref:
                parts.append(f"only in first: {sorted(only_ref)}")
            if only_ds:
                parts.append(f"only in dataset {i}: {sorted(only_ds)}")
            raise ValueError(f"Column mismatch — {'; '.join(parts)}")

    return concatenate_datasets(list(datasets))


def add_string_label_column(
    dataset_info: DatasetInfo,
    dataset: Dataset,
) -> Dataset:
    """Return a new Dataset with an extra column containing the string form of the label column.

    The output column name is given by dataset_info.string_labels_column_name.

    - If dataset_info.label_column_name is a ClassLabel feature, values are converted via int2str().
    - Otherwise, values are cast to str (useful if ClassLabel metadata was lost).
    """
    if dataset_info.label_column_name not in dataset.column_names:
        raise KeyError(
            f"Column '{dataset_info.label_column_name}' not found. Available columns: {dataset.column_names}"
        )

    if dataset_info.string_labels_column_name in dataset.column_names:
        raise ValueError(f"Column '{dataset_info.string_labels_column_name}' already exists. Choose a different name.")

    feature: Any = dataset.features.get(dataset_info.label_column_name)
    if feature is None:
        raise KeyError(
            f"No feature metadata found for column '{dataset_info.label_column_name}'. "
            "Was this Dataset constructed without features?"
        )

    def _to_str(x: Any) -> str | None:
        if x is None:
            return None
        # Many datasets store ClassLabel values as ints (sometimes numpy ints).
        try:
            return feature.int2str(int(x))  # type: ignore
        except Exception:
            return str(x)

    # Preferred path: ClassLabel mapping.
    if isinstance(feature, ClassLabel):

        def _map_a(batch: dict[str, list[Any]]) -> dict[str, list[str | None]]:
            labels = batch[dataset_info.label_column_name]
            return {dataset_info.string_labels_column_name: [_to_str(v) for v in labels]}

        return dataset.map(_map_a, batched=True)

    # Fallback: cast to str (handles string columns and any other type, including cases
    # where ClassLabel metadata was lost).
    def _map_b(batch: dict[str, list[Any]]) -> dict[str, list[str | None]]:
        labels = batch[dataset_info.label_column_name]
        return {dataset_info.string_labels_column_name: [None if v is None else str(v) for v in labels]}

    return dataset.map(_map_b, batched=True)


def add_training_column(
    dataset_info: DatasetInfo,
    training_info: TrainingInfo,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Return a new Dataset with a column containing the chat-template-formatted training text.

    The output column (dataset_info.training_column_name) contains the full conversation:
      - system: training_info.system_prompt
      - user:   dataset_info.content_column_name
      - assistant: dataset_info.string_labels_column_name  (added by add_string_label_column)

    Raises:
        KeyError:   if content_column_name or string_labels_column_name is missing.
        ValueError: if training_column_name already exists.
    """
    for col in (dataset_info.content_column_name, dataset_info.string_labels_column_name):
        if col not in dataset.column_names:
            raise KeyError(f"Column '{col}' not found. Available columns: {dataset.column_names}")
    if dataset_info.training_column_name in dataset.column_names:
        raise ValueError(
            f"Column '{dataset_info.training_column_name}' already exists. Choose a different new_column_prefix."
        )

    system_prompt = training_info.system_prompt

    def _apply_template(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        results = []
        for content, label in zip(
            batch[dataset_info.content_column_name],
            batch[dataset_info.string_labels_column_name],
            strict=True,
        ):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
                {"role": "assistant", "content": label},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            results.append(text)
        return {dataset_info.training_column_name: results}

    return dataset.map(_apply_template, batched=True)


def rebalance_minority_class(
    dataset_info: DatasetInfo,
    dataset: Dataset,
    target_minority_percent: float,
    tolerance: float,
    seed: int = 42,
) -> Dataset:
    """Undersample the majority class so the minority class represents approximately the target percentage.

    Only supports binary classification. For datasets with more than two classes the minority class
    is identified by row count and all other classes are treated as a single majority block, which
    will produce incorrect per-class proportions.

    Args:
        dataset_info: Metadata identifying the label column.
        dataset: The source dataset.
        target_minority_percent: Desired fraction (0–1) for the minority class.
        tolerance: Acceptable absolute deviation from the target (e.g. 0.05 for ±5%).
        seed: Random seed for reproducible sampling.

    Returns:
        A rebalanced Dataset where the minority class is within tolerance of the target.

    Raises:
        ValueError: If inputs are invalid or the target cannot be achieved.
    """
    if not 0.0 < target_minority_percent < 1.0:
        raise ValueError(f"target_minority_percent must be between 0 and 1 exclusive, got {target_minority_percent}")
    if tolerance <= 0.0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    if dataset_info.label_column_name not in dataset.column_names:
        raise KeyError(
            f"Column '{dataset_info.label_column_name}' not found. Available columns: {dataset.column_names}"
        )

    labels: list[Any] = dataset[dataset_info.label_column_name]
    counts: dict[Any, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    minority_class = min(counts, key=counts.__getitem__)
    minority_count = counts[minority_class]
    current_percent = minority_count / len(dataset)

    if abs(current_percent - target_minority_percent) <= tolerance:
        return dataset

    target_non_minority = round(minority_count * (1.0 - target_minority_percent) / target_minority_percent)
    non_minority_indices = [i for i, label in enumerate(labels) if label != minority_class]

    if target_non_minority >= len(non_minority_indices):
        raise ValueError(
            f"Cannot achieve target {target_minority_percent:.1%} ± {tolerance:.1%}: "
            f"minority class '{minority_class}' has {minority_count} rows but would need "
            f"exactly {target_non_minority} non-minority rows (have {len(non_minority_indices)})"
        )

    rng = _random.Random(seed)
    sampled_non_minority = rng.sample(non_minority_indices, target_non_minority)

    minority_indices = [i for i, label in enumerate(labels) if label == minority_class]
    rebalanced = dataset.select(sorted(minority_indices + sampled_non_minority))

    result_percent = minority_count / len(rebalanced)
    if abs(result_percent - target_minority_percent) > tolerance:
        raise AssertionError(
            f"Internal error: rebalancing landed at {result_percent:.2%} which is outside "
            f"target {target_minority_percent:.1%} ± {tolerance:.1%}"
        )

    return rebalanced


def validate_training_column(
    dataset_info: DatasetInfo,
    training_info: TrainingInfo,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_info: ChatTemplateInfo,
) -> None:
    """Validate that every row of the training column is correctly formatted.

    Checks (all applied to every row; violations are collected and reported together):
      1. Training column exists in the dataset.
      2. No null or empty-string values.
      3. EOS token present (when tokenizer.eos_token is not None).
      4. Response separator present (when train_on_outputs_only=True) — absence causes
         train_on_responses_only() to mask all tokens to -100, producing NaN loss.
      5. Instruction separator present (when train_on_outputs_only=True).
      6. At least one non-whitespace character follows the response separator
         (when train_on_outputs_only=True) — guards against truncated/empty labels.

    Raises:
        KeyError:   if training_column_name is not in the dataset.
        ValueError: if any rows fail validation, with a count summary per check.
    """
    col_name = dataset_info.training_column_name
    if col_name not in dataset.column_names:
        raise KeyError(f"Training column '{col_name}' not found. Available columns: {dataset.column_names}")

    eos_token = tokenizer.eos_token
    check_eos = eos_token is not None
    check_separators = training_info.train_on_outputs_only
    response_sep = chat_template_info.response_separator
    instruction_sep = chat_template_info.instruction_separator

    null_empty_count = 0
    eos_count = 0
    response_sep_count = 0
    instruction_sep_count = 0
    empty_label_count = 0

    for value in dataset[col_name]:
        if value is None or value == "":
            null_empty_count += 1
            continue  # remaining checks require a non-null string
        if check_eos and eos_token not in value:
            eos_count += 1
        if check_separators:
            if response_sep not in value:
                response_sep_count += 1
            else:
                last_pos = value.rfind(response_sep)
                after = value[last_pos + len(response_sep) :]
                if not after.strip():
                    empty_label_count += 1
            if instruction_sep not in value:
                instruction_sep_count += 1

    violations: list[str] = []
    total = len(dataset)
    if null_empty_count > 0:
        violations.append(f"  null/empty values: {null_empty_count}/{total} rows")
    if eos_count > 0:
        violations.append(f"  missing EOS token: {eos_count}/{total} rows")
    if response_sep_count > 0:
        violations.append(f"  missing response separator: {response_sep_count}/{total} rows")
    if instruction_sep_count > 0:
        violations.append(f"  missing instruction separator: {instruction_sep_count}/{total} rows")
    if empty_label_count > 0:
        violations.append(f"  no label content after response separator: {empty_label_count}/{total} rows")

    if violations:
        raise ValueError("Training column validation failed:\n" + "\n".join(violations))


def prep_classification_dataset_for_training(
    dataset_info: DatasetInfo,
    training_info: TrainingInfo,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    chat_template_info: ChatTemplateInfo,
    filter_long_content: bool = True,
) -> Dataset:
    return_set: Dataset = add_string_label_column(dataset_info, dataset)
    return_set = add_training_column(dataset_info, training_info, return_set, tokenizer)
    if filter_long_content:
        return_set = filter_by_sequence_length(
            dataset_info, return_set, tokenizer, training_info.max_sequence_length - 5
        )
    validate_training_column(dataset_info, training_info, return_set, tokenizer, chat_template_info)
    return return_set


def add_eval_column(
    dataset_info: DatasetInfo,
    training_info: TrainingInfo,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Return a new Dataset with a column containing the chat-template-formatted eval prompt.

    Unlike add_training_column, the assistant turn is omitted and add_generation_prompt=True
    is used so the formatted string ends with the response-start token. This is the prompt
    fed to model.generate() during post-training evaluation.

    Raises:
        KeyError:   if content_column_name is missing.
        ValueError: if evaluation_instructions_column_name already exists.
    """
    if dataset_info.content_column_name not in dataset.column_names:
        raise KeyError(
            f"Column '{dataset_info.content_column_name}' not found. Available columns: {dataset.column_names}"
        )
    if dataset_info.evaluation_instructions_column_name in dataset.column_names:
        raise ValueError(
            f"Column '{dataset_info.evaluation_instructions_column_name}' already exists. "
            "Choose a different new_column_prefix."
        )

    system_prompt = training_info.system_prompt

    def _apply_template(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        results = []
        for content in batch[dataset_info.content_column_name]:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            results.append(text)
        return {dataset_info.evaluation_instructions_column_name: results}

    return dataset.map(_apply_template, batched=True)


def label_distribution(
    dataset_info: DatasetInfo,
    dataset: Dataset,
) -> dict[str, float]:
    """Return the fraction of rows belonging to each label.

    Uses string_labels_column_name if present, otherwise falls back to
    label_column_name (values are cast to str).

    Raises:
        KeyError: if neither column exists in the dataset.
    """
    if dataset_info.string_labels_column_name in dataset.column_names:
        col = dataset_info.string_labels_column_name
    elif dataset_info.label_column_name in dataset.column_names:
        col = dataset_info.label_column_name
    else:
        raise KeyError(
            f"Neither '{dataset_info.string_labels_column_name}' nor "
            f"'{dataset_info.label_column_name}' found in dataset columns: {dataset.column_names}"
        )

    labels: list[Any] = dataset[col]
    total = len(labels)
    counts: dict[str, int] = {}
    for label in labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1

    return {key: count / total for key, count in counts.items()}


def log_dataset(dataset: Dataset, logger: LoggingProtocol, row_count: int = 1) -> None:
    for i in range(min(row_count, len(dataset))):
        row = dataset[i]
        logger.report_table_message(row)
        logger.add_break()
