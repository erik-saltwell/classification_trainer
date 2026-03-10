from __future__ import annotations

import pytest
from pydantic import ValidationError

from classification_trainer.configuration import (
    LRSchedulerType,
    OptimizerType,
    SFTParameters,
    TrainingInfo,
)
from classification_trainer.utils import FragmentID

# ---------------------------------------------------------------------------
# SFTParameters – standalone
# ---------------------------------------------------------------------------


def test_sft_parameters_defaults() -> None:
    p = SFTParameters()
    assert p.rank == 16
    assert p.alpha_multiplier == 2
    assert p.use_projection_modules is False
    assert p.lora_dropout == 0.0
    assert p.warmup_ratio == 0.05
    assert p.learning_rate == 2e-4
    assert p.optim == OptimizerType.ADAMW_TORCH_8BIT
    assert p.weight_decay == 0.01
    assert p.lr_scheduler_type == LRSchedulerType.LINEAR
    assert p.use_loftq is False
    assert p.loftq_bits == 4
    assert p.loftq_iter == 1
    assert p.rslora is False


def test_sft_parameters_from_dict() -> None:
    p = SFTParameters(rank=32, optim=OptimizerType.ADAMW_TORCH, lr_scheduler_type=LRSchedulerType.COSINE)
    assert p.rank == 32
    assert p.optim == OptimizerType.ADAMW_TORCH
    assert p.lr_scheduler_type == LRSchedulerType.COSINE


def test_sft_parameters_invalid_optim_raises() -> None:
    with pytest.raises(ValidationError):
        SFTParameters(optim="not_a_real_optimizer")  # type: ignore[arg-type]


def test_sft_parameters_invalid_lr_scheduler_raises() -> None:
    with pytest.raises(ValidationError):
        SFTParameters(lr_scheduler_type="not_a_scheduler")  # type: ignore[arg-type]


def test_sft_parameters_is_frozen() -> None:
    p = SFTParameters()
    with pytest.raises(ValidationError):
        p.rank = 99


def test_optimizer_type_str_equality() -> None:
    assert OptimizerType.ADAMW_TORCH == "adamw_torch"
    assert OptimizerType.STABLE_ADAMW == "stable_adamw"


def test_lr_scheduler_type_str_equality() -> None:
    assert LRSchedulerType.COSINE == "cosine"
    assert LRSchedulerType.WARMUP_STABLE_DECAY == "warmup_stable_decay"


def test_all_lr_scheduler_values_are_valid() -> None:
    for v in LRSchedulerType:
        p = SFTParameters(lr_scheduler_type=v)
        assert p.lr_scheduler_type == v


def test_representative_optimizer_values_are_valid() -> None:
    spot_checks = [
        OptimizerType.ADAMW_TORCH,
        OptimizerType.PAGED_ADAMW_8BIT,
        OptimizerType.STABLE_ADAMW,
        OptimizerType.SCHEDULE_FREE_SGD,
        OptimizerType.GALORE_ADAFACTOR_LAYERWISE,
        OptimizerType.LION_8BIT,
    ]
    for optim in spot_checks:
        p = SFTParameters(optim=optim)
        assert p.optim == optim


# ---------------------------------------------------------------------------
# to_dict / from_dict (wandb serialization)
# ---------------------------------------------------------------------------


def test_to_dict_returns_plain_strings() -> None:
    p = SFTParameters(optim=OptimizerType.ADAMW_BNB_8BIT, lr_scheduler_type=LRSchedulerType.COSINE)
    d = p.to_dict()
    assert d["optim"] == "adamw_bnb_8bit"
    assert d["lr_scheduler_type"] == "cosine"
    assert isinstance(d["optim"], str)
    assert isinstance(d["lr_scheduler_type"], str)


def test_to_dict_contains_all_fields() -> None:
    p = SFTParameters()
    d = p.to_dict()
    expected_keys = {
        "rank",
        "alpha_multiplier",
        "use_projection_modules",
        "lora_dropout",
        "warmup_ratio",
        "learning_rate",
        "optim",
        "weight_decay",
        "lr_scheduler_type",
        "use_loftq",
        "loftq_bits",
        "loftq_iter",
        "rslora",
    }
    assert d.keys() == expected_keys


def test_from_dict_roundtrip() -> None:
    original = SFTParameters(rank=32, optim=OptimizerType.PAGED_ADAMW_8BIT, lr_scheduler_type=LRSchedulerType.COSINE)
    d = original.to_dict()
    restored = SFTParameters.from_dict(d)
    assert restored == original


def test_from_dict_invalid_optim_raises() -> None:
    with pytest.raises(ValidationError):
        SFTParameters.from_dict({"optim": "not_valid"})


# ---------------------------------------------------------------------------
# TrainingInfo integration
# ---------------------------------------------------------------------------

_MINIMAL: dict[str, object] = {
    "model_name": "test-model",
    "hugging_face_user_name": "eriksalt",
    "system_prompt_name": FragmentID.RPG_POST_CLASSIFICATION_PROMPT,
    "base_model": "qwen2.5-0.5b-instruct",
    "inference": "simple-classification",
    "model_card_description": "A test model.",
    "training_length_type": "epoch",
    "training_length": 1.0,
    "max_sequence_length": 1536,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
}


def test_training_info_without_sft_parameters_uses_defaults() -> None:
    info = TrainingInfo.model_validate(_MINIMAL)
    assert isinstance(info.sft_parameters, SFTParameters)
    assert info.sft_parameters.rank == 16
    assert info.sft_parameters.optim == OptimizerType.ADAMW_TORCH_8BIT


def test_training_info_with_sft_parameters_dict() -> None:
    data = {**_MINIMAL, "sft_parameters": {"rank": 64, "lr_scheduler_type": "cosine"}}
    info = TrainingInfo.model_validate(data)
    assert info.sft_parameters.rank == 64
    assert info.sft_parameters.lr_scheduler_type == LRSchedulerType.COSINE


def test_training_info_sft_parameters_is_sft_parameters_instance() -> None:
    data = {**_MINIMAL, "sft_parameters": {"rank": 32}}
    info = TrainingInfo.model_validate(data)
    assert isinstance(info.sft_parameters, SFTParameters)


def test_training_info_invalid_sft_optim_raises() -> None:
    data = {**_MINIMAL, "sft_parameters": {"optim": "bad_optimizer"}}
    with pytest.raises(ValidationError):
        TrainingInfo.model_validate(data)


def test_training_info_each_sft_parameter_independent() -> None:
    # Ensure two TrainingInfo instances don't share the same SFTParameters object
    a = TrainingInfo.model_validate(_MINIMAL)
    b = TrainingInfo.model_validate(_MINIMAL)
    assert a.sft_parameters is not b.sft_parameters
