from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value

from classification_trainer.configuration import ChatTemplateInfo, DatasetInfo
from classification_trainer.helpers.dataset_helper import (
    add_eval_column,
    add_string_label_column,
    add_training_column,
    make_stress_split,
    rebalance_minority_class,
    split_dataset,
    take,
    union_datasets,
    validate_training_column,
)

# ─── shared fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def info() -> DatasetInfo:
    return DatasetInfo(
        huggingface_name="owner/myds",
        content_column_name="text",
        label_column_name="label",
        new_column_prefix="ds_",
        training_split_name="train",
        positive_case="pos",
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
    assert "only in dataset 2" in msg


def test_union_three_datasets() -> None:
    ds_a = Dataset.from_dict({"text": [f"a{i}" for i in range(3)], "label": list(range(3))})
    ds_b = Dataset.from_dict({"text": [f"b{i}" for i in range(3)], "label": list(range(3))})
    ds_c = Dataset.from_dict({"text": [f"c{i}" for i in range(3)], "label": list(range(3))})
    result = union_datasets(ds_a, ds_b, ds_c)
    assert len(result) == 9
    assert set(result.column_names) == {"text", "label"}


def test_union_third_dataset_mismatch_raises() -> None:
    ds_a = Dataset.from_dict({"text": ["a"], "label": [0]})
    ds_b = Dataset.from_dict({"text": ["b"], "label": [1]})
    ds_c = Dataset.from_dict({"text": ["c"]})
    with pytest.raises(ValueError, match="Column mismatch"):
        union_datasets(ds_a, ds_b, ds_c)


def test_union_too_few_datasets_raises() -> None:
    ds = Dataset.from_dict({"text": ["a"]})
    with pytest.raises(ValueError):
        union_datasets(ds)


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
    ds = binary_dataset.add_column(info.string_labels_column_name, ["x"] * len(binary_dataset), new_fingerprint="test")
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
    assert result_info.test_split_name == "test"
    assert result_info.validation_split_name == "validation"


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


# ─── add_training_column ─────────────────────────────────────────────────────


@pytest.fixture
def mock_training_info() -> MagicMock:
    ti = MagicMock()
    ti.system_prompt = "You are a classifier."
    return ti


@pytest.fixture
def mock_template_tokenizer() -> MagicMock:
    tok = MagicMock()

    def _apply(messages: list[dict], tokenize: bool, add_generation_prompt: bool) -> str:
        sys = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), None)
        base = f"<sys>{sys}</sys><user>{user}</user>"
        if asst_msg is not None:
            return base + f"<asst>{asst_msg}</asst>"
        if add_generation_prompt:
            return base + "<gen>"
        return base

    tok.apply_chat_template.side_effect = _apply
    return tok


@pytest.fixture
def labelled_dataset(info: DatasetInfo) -> Dataset:
    """Dataset that already has string_labels (as produced by add_string_label_column)."""
    return Dataset.from_dict(
        {
            info.content_column_name: ["hello world", "foo bar", "baz"],
            info.string_labels_column_name: ["pos", "neg", "pos"],
        }
    )


def test_add_training_column_creates_column(
    info: DatasetInfo, mock_training_info: MagicMock, labelled_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    result = add_training_column(info, mock_training_info, labelled_dataset, mock_template_tokenizer)
    assert info.training_column_name in result.column_names


def test_add_training_column_contains_system_prompt(
    info: DatasetInfo, mock_training_info: MagicMock, labelled_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    result = add_training_column(info, mock_training_info, labelled_dataset, mock_template_tokenizer)
    for text in result[info.training_column_name]:
        assert mock_training_info.system_prompt in text


def test_add_training_column_content_and_label_mapped_correctly(
    info: DatasetInfo, mock_training_info: MagicMock, labelled_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    result = add_training_column(info, mock_training_info, labelled_dataset, mock_template_tokenizer)
    rows = result[info.training_column_name]
    assert "<user>hello world</user>" in rows[0]
    assert "<asst>pos</asst>" in rows[0]
    assert "<user>foo bar</user>" in rows[1]
    assert "<asst>neg</asst>" in rows[1]


def test_add_training_column_missing_content_col_raises(
    info: DatasetInfo, mock_training_info: MagicMock, mock_template_tokenizer: MagicMock
) -> None:
    ds = Dataset.from_dict({info.string_labels_column_name: ["pos"]})
    with pytest.raises(KeyError, match=info.content_column_name):
        add_training_column(info, mock_training_info, ds, mock_template_tokenizer)


def test_add_training_column_missing_string_labels_col_raises(
    info: DatasetInfo, mock_training_info: MagicMock, mock_template_tokenizer: MagicMock
) -> None:
    ds = Dataset.from_dict({info.content_column_name: ["hello"]})
    with pytest.raises(KeyError, match=info.string_labels_column_name):
        add_training_column(info, mock_training_info, ds, mock_template_tokenizer)


def test_add_training_column_already_exists_raises(
    info: DatasetInfo, mock_training_info: MagicMock, labelled_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    ds = labelled_dataset.add_column(info.training_column_name, ["x"] * len(labelled_dataset), new_fingerprint="test")
    with pytest.raises(ValueError, match=info.training_column_name):
        add_training_column(info, mock_training_info, ds, mock_template_tokenizer)


# ─── add_eval_column ─────────────────────────────────────────────────────────


@pytest.fixture
def content_only_dataset(info: DatasetInfo) -> Dataset:
    """Dataset with only the content column (no label columns yet)."""
    return Dataset.from_dict({info.content_column_name: ["hello world", "foo bar", "baz"]})


def test_add_eval_column_creates_column(
    info: DatasetInfo, mock_training_info: MagicMock, content_only_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    result = add_eval_column(info, mock_training_info, content_only_dataset, mock_template_tokenizer)
    assert info.evaluation_instructions_column_name in result.column_names


def test_add_eval_column_contains_system_prompt_and_content(
    info: DatasetInfo, mock_training_info: MagicMock, content_only_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    result = add_eval_column(info, mock_training_info, content_only_dataset, mock_template_tokenizer)
    rows = result[info.evaluation_instructions_column_name]
    assert mock_training_info.system_prompt in rows[0]
    assert "hello world" in rows[0]
    assert "foo bar" in rows[1]


def test_add_eval_column_no_assistant_turn(
    info: DatasetInfo, mock_training_info: MagicMock, content_only_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    result = add_eval_column(info, mock_training_info, content_only_dataset, mock_template_tokenizer)
    for text in result[info.evaluation_instructions_column_name]:
        assert "<asst>" not in text
        assert "<gen>" in text  # add_generation_prompt=True sentinel


def test_add_eval_column_uses_add_generation_prompt(
    info: DatasetInfo, mock_training_info: MagicMock, content_only_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    add_eval_column(info, mock_training_info, content_only_dataset, mock_template_tokenizer)
    call_kwargs = mock_template_tokenizer.apply_chat_template.call_args_list[0].kwargs
    assert call_kwargs.get("add_generation_prompt") is True
    assert call_kwargs.get("tokenize") is False


def test_add_eval_column_missing_content_col_raises(
    info: DatasetInfo, mock_training_info: MagicMock, mock_template_tokenizer: MagicMock
) -> None:
    ds = Dataset.from_dict({"other": ["x"]})
    with pytest.raises(KeyError, match=info.content_column_name):
        add_eval_column(info, mock_training_info, ds, mock_template_tokenizer)


def test_add_eval_column_already_exists_raises(
    info: DatasetInfo, mock_training_info: MagicMock, content_only_dataset: Dataset, mock_template_tokenizer: MagicMock
) -> None:
    ds = content_only_dataset.add_column(
        info.evaluation_instructions_column_name, ["x"] * len(content_only_dataset), new_fingerprint="test"
    )
    with pytest.raises(ValueError, match=info.evaluation_instructions_column_name):
        add_eval_column(info, mock_training_info, ds, mock_template_tokenizer)


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


# ─── validate_training_column ─────────────────────────────────────────────────

_INST_SEP = "<|im_start|>user\n"
_RESP_SEP = "<|im_start|>assistant\n"
_EOS = "<|im_end|>"


def _chatml_row(content: str, label: str) -> str:
    return f"<|im_start|>system\nYou are a classifier.{_EOS}\n{_INST_SEP}{content}{_EOS}\n{_RESP_SEP}{label}{_EOS}"


@pytest.fixture
def chatml_template_info() -> ChatTemplateInfo:
    return ChatTemplateInfo(instruction_separator=_INST_SEP, response_separator=_RESP_SEP)


@pytest.fixture
def mock_eos_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.eos_token = _EOS
    return tok


@pytest.fixture
def mock_training_info_outputs_only() -> MagicMock:
    ti = MagicMock()
    ti.train_on_outputs_only = True
    return ti


@pytest.fixture
def mock_training_info_no_outputs_only() -> MagicMock:
    ti = MagicMock()
    ti.train_on_outputs_only = False
    return ti


@pytest.fixture
def valid_training_dataset(info: DatasetInfo) -> Dataset:
    rows = [_chatml_row("hello world", "pos"), _chatml_row("foo bar", "neg"), _chatml_row("baz", "pos")]
    return Dataset.from_dict({info.training_column_name: rows})


def test_validate_happy_path(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    valid_training_dataset: Dataset,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    # Should complete without raising
    validate_training_column(
        info, mock_training_info_outputs_only, valid_training_dataset, mock_eos_tokenizer, chatml_template_info
    )


def test_validate_missing_training_column_raises_key_error(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    ds = Dataset.from_dict({"other_col": ["hello"]})
    with pytest.raises(KeyError):
        validate_training_column(info, mock_training_info_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)


def test_validate_null_value_raises(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    ds = Dataset.from_dict({info.training_column_name: [_chatml_row("a", "pos"), None]})
    with pytest.raises(ValueError, match="null/empty"):
        validate_training_column(info, mock_training_info_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)


def test_validate_empty_string_raises(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    ds = Dataset.from_dict({info.training_column_name: [_chatml_row("a", "pos"), ""]})
    with pytest.raises(ValueError, match="null/empty"):
        validate_training_column(info, mock_training_info_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)


def test_validate_missing_response_separator_raises(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    # Row without the response separator
    bad_row = f"<|im_start|>system\nYou are a classifier.{_EOS}\n{_INST_SEP}hello{_EOS}\nno-separator-here{_EOS}"
    ds = Dataset.from_dict({info.training_column_name: [bad_row]})
    with pytest.raises(ValueError, match="response separator"):
        validate_training_column(info, mock_training_info_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)


def test_validate_missing_instruction_separator_raises(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    bad_row = f"<|im_start|>system\nYou are a classifier.{_EOS}\nno-user-sep{_EOS}\n{_RESP_SEP}pos{_EOS}"
    ds = Dataset.from_dict({info.training_column_name: [bad_row]})
    with pytest.raises(ValueError, match="instruction separator"):
        validate_training_column(info, mock_training_info_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)


def test_validate_nothing_after_response_separator_raises(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    # Response separator present but only whitespace follows
    bad_row = f"{_INST_SEP}hello{_EOS}\n{_RESP_SEP}   \n"
    ds = Dataset.from_dict({info.training_column_name: [bad_row]})
    with pytest.raises(ValueError, match="label content"):
        validate_training_column(info, mock_training_info_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)


def test_validate_missing_eos_raises(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    # Row without the EOS token
    bad_row = f"{_INST_SEP}hello\n{_RESP_SEP}pos"
    ds = Dataset.from_dict({info.training_column_name: [bad_row]})
    with pytest.raises(ValueError, match="EOS"):
        validate_training_column(info, mock_training_info_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)


def test_validate_train_on_outputs_only_false_skips_separator_checks(
    info: DatasetInfo,
    mock_training_info_no_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    # Row with EOS but no separators — should pass when train_on_outputs_only=False
    row = f"some text without separators {_EOS}"
    ds = Dataset.from_dict({info.training_column_name: [row]})
    validate_training_column(info, mock_training_info_no_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)


def test_validate_eos_token_none_skips_eos_check(
    info: DatasetInfo,
    mock_training_info_no_outputs_only: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    tok = MagicMock()
    tok.eos_token = None
    # Row without any EOS — passes because eos_token is None and separators not checked
    ds = Dataset.from_dict({info.training_column_name: ["just some text"]})
    validate_training_column(info, mock_training_info_no_outputs_only, ds, tok, chatml_template_info)


def test_validate_multiple_violations_reported_together(
    info: DatasetInfo,
    mock_training_info_outputs_only: MagicMock,
    mock_eos_tokenizer: MagicMock,
    chatml_template_info: ChatTemplateInfo,
) -> None:
    # Row missing both EOS and response separator (and therefore instruction separator too)
    bad_row = "plain text with no special tokens"
    ds = Dataset.from_dict({info.training_column_name: [bad_row]})
    with pytest.raises(ValueError) as exc_info:
        validate_training_column(info, mock_training_info_outputs_only, ds, mock_eos_tokenizer, chatml_template_info)
    msg = str(exc_info.value)
    assert "EOS" in msg
    assert "response separator" in msg
    assert "instruction separator" in msg
