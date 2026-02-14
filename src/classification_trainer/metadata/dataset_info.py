"""Dataset metadata definitions and lookup utilities."""

from dataclasses import dataclass
from enum import StrEnum


class DatasetName(StrEnum):
    """Enum of supported HuggingFace dataset identifiers."""

    REDDIT_RPG_POST_CLASSIFICATION = "eriksalt/reddit-rpg-rules-question-classification"
    IMDB_TEST = "stanfordnlp/imdb"
    NONE = "none"


@dataclass
class DatasetInfo:
    """Column and split naming conventions for a specific dataset.

    Attributes:
        content_column_name: Name of the column containing the input text.
        training_column_name: Name of the column used during training.
        label_column_name: Name of the column containing numeric labels.
        string_label_column_name: Name of the column containing human-readable string labels.
        prediction_column_name: Name of the column containing model predictions.
        training_split_name: Name of the training data split.
        test_split_name: Name of the test data split.
        eval_split_name: Name of the evaluation data split.
    """

    content_column_name: str
    training_column_name: str
    label_column_name: str
    string_label_column_name: str
    prediction_column_name: str
    training_split_name: str
    test_split_name: str
    eval_split_name: str


_dataset_info: dict[DatasetName, DatasetInfo] = {
    DatasetName.IMDB_TEST: DatasetInfo(
        content_column_name="text",
        training_column_name="train",
        label_column_name="label",
        string_label_column_name="str_label",
        prediction_column_name="prediction",
        training_split_name="train",
        test_split_name="test",
        eval_split_name="eval",
    ),
    DatasetName.REDDIT_RPG_POST_CLASSIFICATION: DatasetInfo(
        content_column_name="content",
        training_column_name="train",
        label_column_name="label",
        string_label_column_name="str_label",
        prediction_column_name="prediction",
        training_split_name="train",
        test_split_name="test",
        eval_split_name="eval",
    ),
}


def get_dataset_info(dataset_name: DatasetName) -> DatasetInfo:
    """Look up the metadata for a given dataset.

    Args:
        dataset_name: The dataset identifier to look up.

    Returns:
        The corresponding DatasetInfo with column and split names.

    Raises:
        KeyError: If the dataset name is not registered.
    """
    if dataset_name not in _dataset_info:
        raise KeyError(dataset_name)
    return _dataset_info[dataset_name]
