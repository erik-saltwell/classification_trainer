from pathlib import Path
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from classification_trainer.configuration import DatasetInfo


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
    new_dataset_info = dataset_info.add_split()
    assert new_dataset_info.training_split_name is not None
    assert new_dataset_info.validation_split_name is not None
    assert new_dataset_info.test_split_name is not None

    if test_percent_of_total + validation_percent_of_total >= 1.0:
        raise ValueError("The sum of test_percent_of_total and val_percent_of_total must be less than 1.0")

    non_train_percent_of_total = test_percent_of_total + validation_percent_of_total
    if non_train_percent_of_total == 0.0:
        raise ValueError("At least one of test_percent_of_total or val_percent_of_total must be greater than 0.0")

    validation_percent_of_non_train = validation_percent_of_total / non_train_percent_of_total

    train_nontrain_sets = dataset.train_test_split(
        test_size=non_train_percent_of_total,
        stratify_by_column=new_dataset_info.label_column_name,
        seed=seed,
    )

    test_val_sets = train_nontrain_sets["test"].train_test_split(
        test_size=validation_percent_of_non_train,
        stratify_by_column=new_dataset_info.label_column_name,
        seed=seed,
    )

    splits = DatasetDict(
        {
            new_dataset_info.training_split_name: train_nontrain_sets["train"],
            new_dataset_info.validation_split_name: test_val_sets["test"],
            new_dataset_info.test_split_name: test_val_sets["train"],
        }
    )
    return splits, new_dataset_info
