from unittest.mock import MagicMock

import pytest
from datasets import Dataset

from classification_trainer.helpers.token_length_helper import (
    TokenLengthData,
    analyze_token_lengths,
    compute_tokens,
    get_percent_samples_within_sequence_length,
)


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.encode.side_effect = lambda text: list(range(len(text)))  # token count == char count
    return tok


@pytest.fixture
def dataset_info() -> MagicMock:
    di = MagicMock()
    di.training_column_name = "text"
    return di


# --- compute_tokens ---


def test_compute_tokens_returns_encode_length(mock_tokenizer):
    mock_tokenizer.encode.side_effect = None
    mock_tokenizer.encode.return_value = [1, 2, 3]
    assert compute_tokens("any", mock_tokenizer) == 3


def test_compute_tokens_empty_string(mock_tokenizer):
    mock_tokenizer.encode.side_effect = None
    mock_tokenizer.encode.return_value = []
    assert compute_tokens("", mock_tokenizer) == 0


def test_compute_tokens_delegates_to_encode(mock_tokenizer):
    mock_tokenizer.encode.side_effect = None
    mock_tokenizer.encode.return_value = []
    compute_tokens("hello", mock_tokenizer)
    mock_tokenizer.encode.assert_called_once_with("hello")


# --- analyze_token_lengths ---


def test_analyze_token_lengths_returns_token_length_data(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * 5] * 10})
    result = analyze_token_lengths(dataset_info, ds, mock_tokenizer)
    assert isinstance(result, TokenLengthData)


def test_analyze_token_lengths_uniform_data(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * 5] * 10})
    result = analyze_token_lengths(dataset_info, ds, mock_tokenizer)
    assert result.p80 == 5
    assert result.p90 == 5
    assert result.p95 == 5
    assert result.p98 == 5
    assert result.p99 == 5
    assert result.p99_5 == 5
    assert result.p100 == 5


def test_analyze_token_lengths_ordering(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * i for i in range(1, 101)]})
    result = analyze_token_lengths(dataset_info, ds, mock_tokenizer)
    assert result.p80 <= result.p90 <= result.p95 <= result.p98 <= result.p99 <= result.p99_5 <= result.p100


def test_analyze_token_lengths_p100_is_max(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * i for i in range(1, 11)]})
    result = analyze_token_lengths(dataset_info, ds, mock_tokenizer)
    assert result.p100 == 10


def test_analyze_token_lengths_correct_column_used(mock_tokenizer):
    di = MagicMock()
    di.training_column_name = "text"
    ds = Dataset.from_dict({"text": ["x"] * 5, "other": ["x" * 50] * 5})
    result = analyze_token_lengths(di, ds, mock_tokenizer)
    assert result.p100 == 1  # not 50


# --- get_percent_samples_within_sequence_length ---


def test_get_percent_all_within(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * 3] * 5})
    assert get_percent_samples_within_sequence_length(dataset_info, ds, mock_tokenizer, 5) == 1.0


def test_get_percent_none_within(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * 10] * 5})
    assert get_percent_samples_within_sequence_length(dataset_info, ds, mock_tokenizer, 5) == 0.0


def test_get_percent_half_within(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * 3, "x" * 3, "x" * 10, "x" * 10]})
    assert get_percent_samples_within_sequence_length(dataset_info, ds, mock_tokenizer, 5) == 0.5


def test_get_percent_boundary_inclusive(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * 5, "x" * 10]})
    assert get_percent_samples_within_sequence_length(dataset_info, ds, mock_tokenizer, 5) == 0.5


def test_get_percent_returns_float(dataset_info, mock_tokenizer):
    ds = Dataset.from_dict({"text": ["x" * 3] * 5})
    result = get_percent_samples_within_sequence_length(dataset_info, ds, mock_tokenizer, 5)
    assert isinstance(result, float)
