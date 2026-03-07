from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from classification_trainer.configuration.sft_parameters import (
    LRSchedulerType,
    OptimizerType,
    SFTParameters,
)


class SweepMethod(StrEnum):
    RANDOM = "random"
    BAYES = "bayes"
    GRID = "grid"


class SweepParameterSpec(BaseModel):
    """Specification for a single sweep parameter.

    Exactly one format must be provided:
    - values: discrete list of values to sample from
    - min + max: continuous range (distribution auto-selected)
    - value: fixed constant held across all trials
    """

    model_config = ConfigDict(frozen=True)

    values: list[int | float | str | bool] | None = None
    min: float | None = None
    max: float | None = None
    value: int | float | str | bool | None = None

    @model_validator(mode="after")
    def _validate_exactly_one_format(self) -> SweepParameterSpec:
        has_values = self.values is not None
        has_range = self.min is not None or self.max is not None
        has_fixed = self.value is not None

        formats_set = sum([has_values, has_range, has_fixed])
        if formats_set == 0:
            raise ValueError(
                "Sweep parameter must specify one of: 'values' (list), 'min'+'max' (range), or 'value' (fixed scalar)"
            )
        if formats_set > 1:
            raise ValueError(
                "Sweep parameter must specify exactly one format: 'values', 'min'+'max', or 'value' — not multiple"
            )

        if has_values and self.values is not None and len(self.values) == 0:
            raise ValueError("Sweep parameter 'values' list must not be empty")

        if has_range:
            if self.min is None or self.max is None:
                raise ValueError("Sweep parameter range requires both 'min' and 'max'")
            if self.min >= self.max:
                raise ValueError(
                    f"Sweep parameter 'min' ({self.min}) must be less than 'max' ({self.max})"
                )

        return self

    def to_wandb_param_spec(self, param_name: str) -> dict[str, Any]:
        """Translate this parameter spec to wandb's sweep config format."""
        if self.values is not None:
            return {"values": list(self.values)}
        if self.min is not None and self.max is not None:
            _LOG_UNIFORM_PARAMS = {"learning_rate"}
            distribution = "log_uniform_values" if param_name in _LOG_UNIFORM_PARAMS else "uniform"
            return {"distribution": distribution, "min": self.min, "max": self.max}
        # fixed value
        return {"value": self.value}


# Domain validation rules per SFTParameters field name.
# Maps field names to validation callables that raise ValueError on invalid values.
_VALID_OPTIM_VALUES = {str(v) for v in OptimizerType}
_VALID_LR_SCHEDULER_VALUES = {str(v) for v in LRSchedulerType}

_DOMAIN_VALIDATORS: dict[str, tuple[str, Any]] = {
    "optim": ("valid OptimizerType string", lambda v: str(v) in _VALID_OPTIM_VALUES),
    "lr_scheduler_type": ("valid LRSchedulerType string", lambda v: str(v) in _VALID_LR_SCHEDULER_VALUES),
    "lora_dropout": ("between 0.0 and 1.0", lambda v: 0.0 <= float(v) <= 1.0),
    "warmup_ratio": ("between 0.0 and 1.0", lambda v: 0.0 <= float(v) <= 1.0),
    "rank": ("a positive integer", lambda v: isinstance(v, int) and v > 0),
    "alpha_multiplier": ("a positive integer", lambda v: isinstance(v, int) and v > 0),
}


def _validate_domain(param_name: str, spec: SweepParameterSpec) -> None:
    """Validate that parameter values fall within the field's valid domain."""
    if param_name not in _DOMAIN_VALIDATORS:
        return
    description, check = _DOMAIN_VALIDATORS[param_name]

    if spec.values is not None:
        for v in spec.values:
            if not check(v):
                raise ValueError(
                    f"Sweep parameter '{param_name}': value {v!r} is invalid — must be {description}"
                )
    if spec.value is not None:
        if not check(spec.value):
            raise ValueError(
                f"Sweep parameter '{param_name}': value {spec.value!r} is invalid — must be {description}"
            )
    if spec.min is not None and spec.max is not None:
        # For range params, validate bounds are within domain where applicable
        if param_name in ("lora_dropout", "warmup_ratio"):
            if not (0.0 <= spec.min <= 1.0 and 0.0 <= spec.max <= 1.0):
                raise ValueError(
                    f"Sweep parameter '{param_name}': min/max must be {description}"
                )


class SweepConfig(BaseModel):
    """Sweep configuration block defining the hyperparameter search space."""

    model_config = ConfigDict(frozen=True)

    method: SweepMethod = SweepMethod.RANDOM
    parameters: dict[str, SweepParameterSpec]

    @model_validator(mode="before")
    @classmethod
    def _coerce_bare_scalars(cls, data: Any) -> Any:
        """Wrap bare scalar values in the parameters dict as SweepParameterSpec(value=X)."""
        if isinstance(data, dict) and "parameters" in data and isinstance(data["parameters"], dict):
            coerced: dict[str, Any] = {}
            for k, v in data["parameters"].items():
                if isinstance(v, (int, float, str, bool)):
                    coerced[k] = {"value": v}
                else:
                    coerced[k] = v
            data = {**data, "parameters": coerced}
        return data

    @model_validator(mode="after")
    def _validate_sweep_config(self) -> SweepConfig:
        if not self.parameters:
            raise ValueError("Sweep block requires at least one parameter in 'parameters'")

        valid_fields = set(SFTParameters.model_fields.keys())
        invalid_keys = set(self.parameters.keys()) - valid_fields
        if invalid_keys:
            raise ValueError(
                f"Unknown sweep parameter(s): {sorted(invalid_keys)}. "
                f"Valid: {sorted(valid_fields)}"
            )

        if self.method == SweepMethod.GRID:
            for name, spec in self.parameters.items():
                if spec.min is not None or spec.max is not None:
                    raise ValueError(
                        f"Grid search requires all parameters to use discrete values. "
                        f"'{name}' uses min/max range"
                    )

        for name, spec in self.parameters.items():
            _validate_domain(name, spec)

        return self

    def to_wandb_sweep_config(
        self,
        sft_parameters: SFTParameters,
        metric_name: str,
        metric_goal: str,
    ) -> dict[str, Any]:
        """Build the full wandb sweep config dict.

        Listed parameters use their sweep spec; unlisted parameters
        use the fixed value from sft_parameters.
        """
        sft_dict = sft_parameters.to_dict()
        wandb_params: dict[str, Any] = {}
        for field_name in SFTParameters.model_fields:
            if field_name in self.parameters:
                wandb_params[field_name] = self.parameters[field_name].to_wandb_param_spec(field_name)
            else:
                wandb_params[field_name] = {"value": sft_dict[field_name]}

        return {
            "method": str(self.method),
            "metric": {"goal": metric_goal, "name": metric_name},
            "parameters": wandb_params,
        }
