from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from classification_trainer.configuration import DatasetInfo
from classification_trainer.helpers.dataset_helper import (
    add_string_label_column,
    make_stress_split,
    rebalance_minority_class,
    split_dataset,
    take,
    union_datasets,
)

# ─── shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def info() -> DatasetInfo:
    return DatasetInfo(
        huggingface_name="owner/myds",
        content_column_name="text",
        label_column_name="label",
        new_column_prefix="ds_",
    )


@pytest.fixture
def binary_dataset() -> Dataset:
    # 100 rows, 50 class-0 / 50 class-1, with ClassLabel feature
    features = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})
    return Dataset.from_dict(
        {"text": [f"row {i}" for i in range(100)], "label": [i % 2 for i in range(100)]},
        features=features,
    )


# ─── take ────────────────────────────────────────────────────────────────────


def test_take_fewer_than_count() -> None:
    ds = Dataset.from_dict({"x": list(range(5))})
    result = take(ds, 3)
    assert len(result) == 3


def test_take_more_than_count() -> None:
    ds = Dataset.from_dict({"x": list(range(5))})
    result = take(ds, 20)
    assert len(result) == 5


def test_take_zero() -> None:
    ds = Dataset.from_dict({"x": list(range(5))})
    result = take(ds, 0)
    assert len(result) == 0


# ─── union_datasets ──────────────────────────────────────────────────────────


def test_union_same_columns() -> None:
    ds_a = Dataset.from_dict({"text": [f"a{i}" for i in range(5)], "label": list(range(5))})
    ds_b = Dataset.from_dict({"text": [f"b{i}" for i in range(5)], "label": list(range(5))})
    result = union_datasets(ds_a, ds_b)
    assert len(result) == 10
    assert set(result.column_names) == {"text", "label"}


def test_union_column_mismatch_raises() -> None:
    ds_a = Dataset.from_dict({"text": ["a"], "extra": [1]})
    ds_b = Dataset.from_dict({"text": ["b"]})
    with pytest.raises(ValueError, match="Column mismatch"):
        union_datasets(ds_a, ds_b)


def test_union_mismatch_message_both_sides() -> None:
    ds_a = Dataset.from_dict({"text": ["a"], "only_a": [1]})
    ds_b = Dataset.from_dict({"text": ["b"], "only_b": [2]})
    with pytest.raises(ValueError) as exc_info:
        union_datasets(ds_a, ds_b)
    msg = str(exc_info.value)
    assert "only in first" in msg
    assert "only in second" in msg


# ─── add_string_label_column ─────────────────────────────────────────────────


def test_add_string_label_class_label(info: DatasetInfo, binary_dataset: Dataset) -> None:
    result = add_string_label_column(info, binary_dataset)
    assert info.string_labels_column_name in result.column_names
    string_labels = result[info.string_labels_column_name]
    assert set(string_labels) == {"neg", "pos"}


def test_add_string_label_non_class_label(info: DatasetInfo) -> None:
    # Plain int column — no ClassLabel feature, fallback to str()
    ds = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
    result = add_string_label_column(info, ds)
    assert info.string_labels_column_name in result.column_names
    assert result[info.string_labels_column_name] == ["0", "1"]


def test_add_string_label_missing_col_raises(info: DatasetInfo) -> None:
    ds = Dataset.from_dict({"text": ["a"]})
    with pytest.raises(KeyError):
        add_string_label_column(info, ds)


def test_add_string_label_col_already_exists_raises(info: DatasetInfo, binary_dataset: Dataset) -> None:
    ds = binary_dataset.add_column(info.string_labels_column_name, ["x"] * len(binary_dataset))
    with pytest.raises(ValueError):
        add_string_label_column(info, ds)


# ─── split_dataset ───────────────────────────────────────────────────────────


def test_split_dataset_already_split_raises(info: DatasetInfo, binary_dataset: Dataset) -> None:
    split_info = info.add_split()
    with pytest.raises(ValueError):
        split_dataset(split_info, binary_dataset, 0.1, 0.1, seed=42)


def test_split_dataset_percents_too_large_raises(info: DatasetInfo, binary_dataset: Dataset) -> None:
    with pytest.raises(ValueError):
        split_dataset(info, binary_dataset, 0.5, 0.5, seed=42)


def test_split_dataset_both_zero_raises(info: DatasetInfo, binary_dataset: Dataset) -> None:
    with pytest.raises(ValueError):
        split_dataset(info, binary_dataset, 0.0, 0.0, seed=42)


def test_split_dataset_happy_path(info: DatasetInfo, binary_dataset: Dataset) -> None:
    result_dict, result_info = split_dataset(info, binary_dataset, 0.1, 0.1, seed=42)
    assert isinstance(result_dict, DatasetDict)
    assert set(result_dict.keys()) == {"train", "validation", "test"}
    total = sum(len(result_dict[k]) for k in result_dict)
    assert total == 100
    assert result_info.is_split is True


# ─── make_stress_split ───────────────────────────────────────────────────────


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.encode.side_effect = lambda text: list(range(len(text)))  # token count == char count
    return tok


@pytest.fixture
def stress_dataset(info: DatasetInfo) -> Dataset:
    """10 rows with varying text lengths in the training column (lengths 1–10)."""
    training_col = info.training_column_name  # "ds_training"
    texts = ["x" * (i + 1) for i in range(10)]
    return Dataset.from_dict({training_col: texts, "label": list(range(10))})


def test_make_stress_split_returns_n_rows(
    info: DatasetInfo, stress_dataset: Dataset, mock_tokenizer: MagicMock
) -> None:
    result = make_stress_split(info, stress_dataset, 3, mock_tokenizer)
    assert len(result) == 3


def test_make_stress_split_longest_first(info: DatasetInfo, stress_dataset: Dataset, mock_tokenizer: MagicMock) -> None:
    result = make_stress_split(info, stress_dataset, 3, mock_tokenizer)
    training_col = info.training_column_name
    lengths = {len(t) for t in result[training_col]}
    assert lengths == {10, 9, 8}


def test_make_stress_split_clamps_to_dataset_size(
    info: DatasetInfo, stress_dataset: Dataset, mock_tokenizer: MagicMock
) -> None:
    result = make_stress_split(info, stress_dataset, 1000, mock_tokenizer)
    assert len(result) == len(stress_dataset)


def test_make_stress_split_no_temp_column(
    info: DatasetInfo, stress_dataset: Dataset, mock_tokenizer: MagicMock
) -> None:
    result = make_stress_split(info, stress_dataset, 3, mock_tokenizer)
    assert "_token_count" not in result.column_names


# ─── rebalance_minority_class ────────────────────────────────────────────────


def test_rebalance_already_balanced_returns_same(info: DatasetInfo, binary_dataset: Dataset) -> None:
    # 50/50 dataset, target=0.5, tol=0.05 → within tolerance, same object returned
    result = rebalance_minority_class(info, binary_dataset, 0.5, 0.05)
    assert result is binary_dataset


def test_rebalance_achieves_target(info: DatasetInfo) -> None:
    # 10 minority (class-0), 90 majority (class-1), target 30% minority
    features = Features({"text": Value("string"), "label": ClassLabel(names=["neg", "pos"])})
    ds = Dataset.from_dict(
        {"text": [f"r{i}" for i in range(100)], "label": [0] * 10 + [1] * 90},
        features=features,
    )
    result = rebalance_minority_class(info, ds, 0.3, 0.05)
    minority_count = sum(1 for v in result["label"] if v == 0)
    minority_pct = minority_count / len(result)
    assert abs(minority_pct - 0.3) <= 0.05


def test_rebalance_invalid_target_raises(info: DatasetInfo, binary_dataset: Dataset) -> None:
    with pytest.raises(ValueError):
        rebalance_minority_class(info, binary_dataset, 0.0, 0.05)
    with pytest.raises(ValueError):
        rebalance_minority_class(info, binary_dataset, 1.0, 0.05)


def test_rebalance_invalid_tolerance_raises(info: DatasetInfo, binary_dataset: Dataset) -> None:
    with pytest.raises(ValueError):
        rebalance_minority_class(info, binary_dataset, 0.5, 0.0)


def test_rebalance_missing_label_col_raises(info: DatasetInfo) -> None:
    ds = Dataset.from_dict({"text": ["a", "b"]})
    with pytest.raises(KeyError):
        rebalance_minority_class(info, ds, 0.3, 0.05)


def test_rebalance_cannot_achieve_raises(info: DatasetInfo) -> None:
    # 5 minority (class-0), 10 majority (class-1), target=0.2
    # needs round(5 * 0.8/0.2)=20 majority rows but only 10 available → ValueError
    ds = Dataset.from_dict(
        {"text": [f"r{i}" for i in range(15)], "label": [0] * 5 + [1] * 10},
    )
    with pytest.raises(ValueError):
        rebalance_minority_class(info, ds, 0.2, 0.05)
