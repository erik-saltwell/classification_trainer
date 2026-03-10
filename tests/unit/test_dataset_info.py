from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from classification_trainer.configuration.dataset_info import DatasetInfo

_MINIMAL: dict[str, Any] = {
    "dataset_name": "myds",
    "hugging_face_user_name": "owner",
    "content_column_name": "text",
    "label_column_name": "label",
    "new_column_prefix": "ds_",
    "training_split_name": "train",
    "positive_case": "pos",
}


# ---------------------------------------------------------------------------
# sequence_lengths — defaults
# ---------------------------------------------------------------------------


def test_sequence_lengths_default() -> None:
    info = DatasetInfo(**_MINIMAL)
    assert info.potential_sequence_lengths == [1024, 1536, 2048]


# ---------------------------------------------------------------------------
# sequence_lengths — valid values
# ---------------------------------------------------------------------------


def test_sequence_lengths_custom_list() -> None:
    info = DatasetInfo(**{**_MINIMAL, "potential_sequence_lengths": [512, 768, 4096]})
    assert info.potential_sequence_lengths == [512, 768, 4096]


def test_sequence_lengths_single_value() -> None:
    info = DatasetInfo(**{**_MINIMAL, "potential_sequence_lengths": [2048]})
    assert info.potential_sequence_lengths == [2048]


def test_sequence_lengths_deduplicates_preserving_order() -> None:
    info = DatasetInfo(**{**_MINIMAL, "potential_sequence_lengths": [2048, 1024, 2048, 512, 1024]})
    assert info.potential_sequence_lengths == [2048, 1024, 512]


# ---------------------------------------------------------------------------
# sequence_lengths — invalid values
# ---------------------------------------------------------------------------


def test_sequence_lengths_empty_list_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        DatasetInfo(**{**_MINIMAL, "potential_sequence_lengths": []})


def test_sequence_lengths_zero_rejected() -> None:
    with pytest.raises(ValidationError, match="positive"):
        DatasetInfo(**{**_MINIMAL, "potential_sequence_lengths": [0]})


def test_sequence_lengths_negative_rejected() -> None:
    with pytest.raises(ValidationError, match="positive"):
        DatasetInfo(**{**_MINIMAL, "potential_sequence_lengths": [-1]})


def test_sequence_lengths_mixed_valid_and_invalid_rejected() -> None:
    with pytest.raises(ValidationError, match="positive"):
        DatasetInfo(**{**_MINIMAL, "potential_sequence_lengths": [1024, -1, 2048]})


# ---------------------------------------------------------------------------
# get_generated_column_names — pre-tokenization columns
# ---------------------------------------------------------------------------


def test_generated_column_names_include_pretokenized_columns() -> None:
    info = DatasetInfo(**_MINIMAL)
    names = list(info.get_generated_column_names())
    for col in ("input_ids", "attention_mask", "labels", "eval_input_ids", "eval_attention_mask"):
        assert col in names, f"Expected '{col}' in generated column names"
