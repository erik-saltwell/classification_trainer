"""Dataset metadata definitions and lookup utilities."""

from enum import StrEnum


class DatasetName(StrEnum):
    """Enum of supported HuggingFace dataset identifiers."""

    REDDIT_RPG_POST_CLASSIFICATION = "eriksalt/reddit-rpg-rules-question-classification"
    IMDB_TEST = "stanfordnlp/imdb"
    NONE = "none"


class DatasetInfo:
    huggingface_name: str  # must have value of nonzero length, and be valiid filesystem name (with no exetensions).
    content_column_name: str  # must be valid huggging face dataset column name
    label_column_name: str  # must be valid huggging face dataset column name
    new_column_prefix: str  # must be valid huggging face dataset column name
    is_split: bool = False  # whether or not this dataset has multiple Dataset splits inside it
    training_split_name: str | None = None  # must be valid hugging face split name
    test_split_name: str | None = None  # must be valid hugging face split name
    eval_split_name: str | None = None  # must be valid hugging face split name
