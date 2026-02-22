import random as _random
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

from classification_trainer.configuration import DatasetInfo, TrainingInfo

from .token_length_helper import compute_tokens
from .tokenizer_helper import apply_chat_template, generate_eos


def load_dataset_from_hf(dataset_info: DatasetInfo) -> DatasetDict:
    return cast(DatasetDict, load_dataset(dataset_info.huggingface_name))


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
    if dataset_info.is_split:
        raise ValueError("Cannot split a dataset that is already split.")
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
    assert new_dataset_info.training_split_name is not None
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


def take(dataset: Dataset, count: int) -> Dataset:
    """Return a new Dataset with at most *count* records from the start."""
    return dataset.select(range(min(count, len(dataset))))


def union_datasets(a: Dataset, b: Dataset) -> Dataset:
    """Concatenate two Datasets after validating they share the same columns.

    Raises:
        ValueError: If the column names differ between the two datasets.
    """
    cols_a = set(a.column_names)
    cols_b = set(b.column_names)
    if cols_a != cols_b:
        only_a = cols_a - cols_b
        only_b = cols_b - cols_a
        parts: list[str] = []
        if only_a:
            parts.append(f"only in first: {sorted(only_a)}")
        if only_b:
            parts.append(f"only in second: {sorted(only_b)}")
        raise ValueError(f"Column mismatch — {'; '.join(parts)}")
    return concatenate_datasets([a, b])


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


RowFormatter = Callable[[str, str], str]


def _apply_template(
    data: Mapping[str, list[Any]],
    format_row: RowFormatter,
    content_column_name: str,
    labels_column_name: str,
    new_column_name: str,
) -> dict[str, list[str]]:
    inputs = data[content_column_name]
    outputs = data[labels_column_name]
    return {new_column_name: [format_row(content, label) for content, label in zip(inputs, outputs, strict=True)]}


def add_training_column(
    dataset_info: DatasetInfo,
    training_info: TrainingInfo,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    eos = generate_eos(tokenizer)

    def format_row(inp: str, out: str) -> str:
        return apply_chat_template(training_info.system_prompt, inp, out, tokenizer, eos)

    return dataset.map(
        lambda data: _apply_template(
            data,
            format_row,
            dataset_info.content_column_name,
            dataset_info.string_labels_column_name,
            dataset_info.training_column_name,
        ),
        batched=True,
    )


def add_eval_column(
    dataset_info: DatasetInfo,
    training_info: TrainingInfo,
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    eos = generate_eos(tokenizer)

    # _out is unused: eval prompts are open-ended (no assistant turn), but RowFormatter
    # requires two arguments because _apply_template always passes both content and label.
    def format_row(inp: str, _out: str) -> str:
        return apply_chat_template(training_info.system_prompt, inp, "", tokenizer, eos)

    return dataset.map(
        lambda data: _apply_template(
            data,
            format_row,
            dataset_info.content_column_name,
            dataset_info.string_labels_column_name,
            dataset_info.evaluation_instructions_column_name,
        ),
        batched=True,
    )
