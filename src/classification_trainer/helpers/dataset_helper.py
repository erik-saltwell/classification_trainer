from pathlib import Path
from typing import cast

from datasets import DatasetDict, load_dataset, load_from_disk

from classification_trainer.configuration import DatasetInfo


def load_dataset_from_hf(dataset_info: DatasetInfo) -> DatasetDict:
    return cast(DatasetDict, load_dataset(dataset_info.huggingface_name))


def load_dataset_from_disk(path: Path) -> DatasetDict:
    return cast(DatasetDict, load_from_disk(str(path)))


def save_dataset_to_disk(dataset: DatasetDict, path: Path) -> None:
    dataset.save_to_disk(str(path))
