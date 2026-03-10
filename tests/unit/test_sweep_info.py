from __future__ import annotations

import pytest
from pydantic import ValidationError

from classification_trainer.configuration.sweep_info import (
    SweepInfo,
    SweepMethod,
    SweepParameterSpec,
)

# ---------------------------------------------------------------------------
# SweepParameterSpec — valid formats
# ---------------------------------------------------------------------------


def test_valid_discrete_list() -> None:
    spec = SweepParameterSpec(values=[8, 16, 32])
    assert spec.values == [8, 16, 32]


def test_valid_min_max_range() -> None:
    spec = SweepParameterSpec(min=1e-5, max=1e-3)
    assert spec.min == 1e-5
    assert spec.max == 1e-3


def test_valid_fixed_value() -> None:
    spec = SweepParameterSpec(value=32)
    assert spec.value == 32


# ---------------------------------------------------------------------------
# SweepParameterSpec — invalid formats
# ---------------------------------------------------------------------------


def test_empty_values_list_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        SweepParameterSpec(values=[])


def test_min_greater_than_max_rejected() -> None:
    with pytest.raises(ValidationError, match="must be less than"):
        SweepParameterSpec(min=1e-3, max=1e-5)


def test_min_equal_to_max_rejected() -> None:
    with pytest.raises(ValidationError, match="must be less than"):
        SweepParameterSpec(min=0.1, max=0.1)


def test_min_without_max_rejected() -> None:
    with pytest.raises(ValidationError, match="both 'min' and 'max'"):
        SweepParameterSpec(min=0.1)


def test_max_without_min_rejected() -> None:
    with pytest.raises(ValidationError, match="both 'min' and 'max'"):
        SweepParameterSpec(max=0.5)


def test_no_format_specified_rejected() -> None:
    with pytest.raises(ValidationError, match="must specify one of"):
        SweepParameterSpec()


def test_multiple_formats_rejected() -> None:
    with pytest.raises(ValidationError, match="exactly one format"):
        SweepParameterSpec(values=[1, 2], value=3)


# ---------------------------------------------------------------------------
# SweepInfo — valid configs
# ---------------------------------------------------------------------------


def _make_config(**overrides: object) -> SweepInfo:
    defaults: dict[str, object] = {
        "sweep_name": "test-sweep",
        "description": "test sweep",
        "run_cap": 10,
        "parameters": {"rank": {"values": [8, 16, 32]}},
    }
    defaults.update(overrides)
    return SweepInfo.model_validate(defaults)


def test_valid_sweep_config() -> None:
    config = _make_config()
    assert config.method == SweepMethod.RANDOM
    assert "rank" in config.parameters


def test_method_defaults_to_random() -> None:
    config = _make_config()
    assert config.method == SweepMethod.RANDOM


# ---------------------------------------------------------------------------
# SweepInfo — invalid configs
# ---------------------------------------------------------------------------


def test_invalid_parameter_name_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown sweep parameter"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"lerning_rate": {"values": [0.01]}},
            }
        )


def test_empty_parameters_dict_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one parameter"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {},
            }
        )


def test_grid_with_continuous_range_rejected() -> None:
    with pytest.raises(ValidationError, match="Grid search requires"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "method": "grid",
                "parameters": {"learning_rate": {"min": 1e-5, "max": 1e-3}},
            }
        )


def test_grid_with_all_discrete_accepted() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "method": "grid",
            "parameters": {"rank": {"values": [8, 16]}, "optim": {"values": ["adamw_bnb_8bit"]}},
        }
    )
    assert config.method == SweepMethod.GRID


# ---------------------------------------------------------------------------
# SweepInfo — domain validation (T006)
# ---------------------------------------------------------------------------


def test_lora_dropout_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError, match="lora_dropout.*invalid"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"lora_dropout": {"values": [1.5]}},
            }
        )


def test_optim_invalid_value_rejected() -> None:
    with pytest.raises(ValidationError, match="optim.*invalid"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"optim": {"values": ["not_real"]}},
            }
        )


def test_rank_negative_rejected() -> None:
    with pytest.raises(ValidationError, match="rank.*invalid"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"rank": {"values": [-1]}},
            }
        )


def test_rank_zero_rejected() -> None:
    with pytest.raises(ValidationError, match="rank.*invalid"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"rank": {"values": [0]}},
            }
        )


def test_warmup_ratio_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError, match="warmup_ratio.*invalid"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"warmup_ratio": {"values": [1.5]}},
            }
        )


def test_lr_scheduler_invalid_value_rejected() -> None:
    with pytest.raises(ValidationError, match="lr_scheduler_type.*invalid"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"lr_scheduler_type": {"values": ["bogus"]}},
            }
        )


def test_valid_domain_values_accepted() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {
                "rank": {"values": [8, 16]},
                "lora_dropout": {"values": [0.0, 0.1]},
                "optim": {"values": ["adamw_bnb_8bit", "sgd"]},
                "lr_scheduler_type": {"values": ["linear", "cosine"]},
                "warmup_ratio": {"values": [0.05, 0.1]},
            },
        }
    )
    assert len(config.parameters) == 5


def test_lora_dropout_range_out_of_domain_rejected() -> None:
    with pytest.raises(ValidationError, match="lora_dropout.*min/max"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"lora_dropout": {"min": 0.0, "max": 1.5}},
            }
        )


def test_lora_dropout_range_valid_accepted() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {"lora_dropout": {"min": 0.0, "max": 0.5}},
        }
    )
    assert config.parameters["lora_dropout"].min == 0.0


def test_fixed_value_domain_validated() -> None:
    with pytest.raises(ValidationError, match="optim.*invalid"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "parameters": {"optim": {"value": "not_real"}},
            }
        )


# ---------------------------------------------------------------------------
# to_wandb_param_spec (T011)
# ---------------------------------------------------------------------------


def test_wandb_discrete_list() -> None:
    spec = SweepParameterSpec(values=[8, 16, 32])
    assert spec.to_wandb_param_spec("rank") == {"values": [8, 16, 32]}


def test_wandb_learning_rate_log_uniform() -> None:
    spec = SweepParameterSpec(min=1e-5, max=1e-3)
    result = spec.to_wandb_param_spec("learning_rate")
    assert result == {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-3}


def test_wandb_weight_decay_uniform() -> None:
    spec = SweepParameterSpec(min=0.0, max=0.1)
    result = spec.to_wandb_param_spec("weight_decay")
    assert result == {"distribution": "uniform", "min": 0.0, "max": 0.1}


def test_wandb_fixed_scalar() -> None:
    spec = SweepParameterSpec(value="adamw_bnb_8bit")
    assert spec.to_wandb_param_spec("optim") == {"value": "adamw_bnb_8bit"}


# ---------------------------------------------------------------------------
# to_wandb_sweep_config (T012)
# ---------------------------------------------------------------------------


def test_wandb_sweep_config_listed_params() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {
                "rank": {"values": [8, 32]},
                "learning_rate": {"min": 1e-5, "max": 1e-3},
            },
        }
    )
    result = config.to_wandb_sweep_config("f1", "maximize")
    assert result["parameters"]["rank"] == {"values": [8, 32]}
    assert result["parameters"]["learning_rate"]["distribution"] == "log_uniform_values"


def test_wandb_sweep_config_only_swept_params_present() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {"rank": {"values": [8, 32]}},
        }
    )
    result = config.to_wandb_sweep_config("f1", "maximize")
    # only the swept field is present — unlisted fields are not sent to wandb
    assert list(result["parameters"].keys()) == ["rank"]
    assert "optim" not in result["parameters"]
    assert "learning_rate" not in result["parameters"]


def test_wandb_sweep_config_method_and_metric() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "method": "bayes",
            "parameters": {"rank": {"values": [8, 16]}},
        }
    )
    result = config.to_wandb_sweep_config("accuracy", "maximize")
    assert result["method"] == "bayes"
    assert result["metric"] == {"goal": "maximize", "name": "accuracy"}


def test_wandb_sweep_config_only_swept_fields_in_parameters() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {
                "rank": {"values": [8]},
                "learning_rate": {"min": 1e-5, "max": 1e-3},
            },
        }
    )
    result = config.to_wandb_sweep_config("f1", "maximize")
    # parameters block contains exactly the swept fields, nothing more
    assert set(result["parameters"].keys()) == {"rank", "learning_rate"}


# ---------------------------------------------------------------------------
# Bare scalar coercion (T014)
# ---------------------------------------------------------------------------


def test_bare_scalar_integer() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {"rank": 32},
        }
    )
    assert config.parameters["rank"].value == 32
    assert config.parameters["rank"].values is None


def test_bare_scalar_string() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {"optim": "adamw_bnb_8bit"},
        }
    )
    assert config.parameters["optim"].value == "adamw_bnb_8bit"


def test_bare_scalar_float() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {"learning_rate": 0.0002},
        }
    )
    assert config.parameters["learning_rate"].value == 0.0002


def test_bare_scalar_boolean() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {"use_projection_modules": False},
        }
    )
    assert config.parameters["use_projection_modules"].value is False


# ---------------------------------------------------------------------------
# Method selection (T015)
# ---------------------------------------------------------------------------


def test_method_bayes_in_wandb_config() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "method": "bayes",
            "parameters": {"rank": {"values": [8, 16]}},
        }
    )
    result = config.to_wandb_sweep_config("f1", "maximize")
    assert result["method"] == "bayes"


def test_method_grid_in_wandb_config() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "method": "grid",
            "parameters": {"rank": {"values": [8, 16]}},
        }
    )
    result = config.to_wandb_sweep_config("f1", "maximize")
    assert result["method"] == "grid"


def test_method_default_random_in_wandb_config() -> None:
    config = SweepInfo.model_validate(
        {
            "sweep_name": "test-sweep",
            "description": "test sweep",
            "run_cap": 10,
            "parameters": {"rank": {"values": [8, 16]}},
        }
    )
    result = config.to_wandb_sweep_config("f1", "maximize")
    assert result["method"] == "random"


def test_grid_with_min_max_raises_at_validation() -> None:
    with pytest.raises(ValidationError, match="Grid search"):
        SweepInfo.model_validate(
            {
                "sweep_name": "test-sweep",
                "description": "test sweep",
                "run_cap": 10,
                "method": "grid",
                "parameters": {"learning_rate": {"min": 1e-5, "max": 1e-3}},
            }
        )


# ---------------------------------------------------------------------------
# Distribution auto-selection (T016)
# ---------------------------------------------------------------------------


def test_distribution_learning_rate_log_uniform() -> None:
    spec = SweepParameterSpec(min=1e-5, max=1e-3)
    result = spec.to_wandb_param_spec("learning_rate")
    assert result["distribution"] == "log_uniform_values"


def test_distribution_weight_decay_uniform() -> None:
    spec = SweepParameterSpec(min=0.0, max=0.1)
    result = spec.to_wandb_param_spec("weight_decay")
    assert result["distribution"] == "uniform"


def test_distribution_lora_dropout_uniform() -> None:
    spec = SweepParameterSpec(min=0.0, max=0.5)
    result = spec.to_wandb_param_spec("lora_dropout")
    assert result["distribution"] == "uniform"


def test_distribution_warmup_ratio_uniform() -> None:
    spec = SweepParameterSpec(min=0.01, max=0.2)
    result = spec.to_wandb_param_spec("warmup_ratio")
    assert result["distribution"] == "uniform"
