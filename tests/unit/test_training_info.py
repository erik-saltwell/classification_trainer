"""Unit tests for TrainingInfo new reference fields and resolution properties."""

from __future__ import annotations

from pathlib import Path

import pytest

from classification_trainer.configuration.training_info import TrainingInfo

# Project root (two levels up from tests/unit/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Minimal valid TrainingInfo dict with the new required fields
# ---------------------------------------------------------------------------

_BASE_DICT: dict = {
    "model_name": "test-classifier",
    "hugging_face_user_name": "testuser",
    "system_prompt_name": "rpg_post_classification_prompt.md",
    "base_model": "qwen2.5-0.5b-instruct",
    "dataset": "imdb",
    "inference": "simple-classification",
    "publishing": None,
    "model_card_description": "A test classifier.",
    "training_length_type": "epoch",
    "training_length": 1.0,
    "max_sequence_length": 1024,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "sft_parameters": {
        "rank": 16,
        "alpha_multiplier": 2,
        "use_projection_modules": False,
        "lora_dropout": 0.0,
        "warmup_ratio": 0.05,
        "learning_rate": 2e-4,
        "optim": "adamw_bnb_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
    },
}


# ---------------------------------------------------------------------------
# T006: New field loading tests
# ---------------------------------------------------------------------------


def test_training_info_loads_new_fields() -> None:
    info = TrainingInfo(**_BASE_DICT)
    assert info.base_model == "qwen2.5-0.5b-instruct"
    assert info.inference == "simple-classification"
    assert info.publishing is None
    assert info.model_card_description == "A test classifier."


def test_training_info_publishing_field_set() -> None:
    data = {**_BASE_DICT, "publishing": "standard-publish"}
    info = TrainingInfo(**data)
    assert info.publishing == "standard-publish"


def test_training_info_model_card_description_is_string() -> None:
    info = TrainingInfo(**_BASE_DICT)
    assert isinstance(info.model_card_description, str)


# ---------------------------------------------------------------------------
# T005: Resolution property tests
# ---------------------------------------------------------------------------


def test_base_model_info_resolves_existing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_PROJECT_ROOT)
    info = TrainingInfo(**_BASE_DICT)
    bm = info.base_model_info
    assert bm.huggingface_name  # loaded successfully


def test_base_model_info_raises_on_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_PROJECT_ROOT)
    data = {**_BASE_DICT, "base_model": "nonexistent-model-xyz"}
    info = TrainingInfo(**data)
    with pytest.raises(FileNotFoundError):
        _ = info.base_model_info


def test_inference_info_resolves_existing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_PROJECT_ROOT)
    info = TrainingInfo(**_BASE_DICT)
    inf = info.inference_info
    assert inf.metrics  # loaded successfully


def test_inference_info_raises_on_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_PROJECT_ROOT)
    data = {**_BASE_DICT, "inference": "nonexistent-inference-xyz"}
    info = TrainingInfo(**data)
    with pytest.raises(FileNotFoundError):
        _ = info.inference_info


def test_publishing_info_returns_none_when_not_set() -> None:
    info = TrainingInfo(**_BASE_DICT)
    assert info.publishing_info is None


def test_publishing_info_raises_on_missing_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(_PROJECT_ROOT)
    data = {**_BASE_DICT, "publishing": "nonexistent-publishing-xyz"}
    info = TrainingInfo(**data)
    with pytest.raises(FileNotFoundError):
        _ = info.publishing_info
