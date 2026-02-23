from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class LRSchedulerType(StrEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"
    COSINE_WITH_MIN_LR = "cosine_with_min_lr"
    COSINE_WARMUP_WITH_MIN_LR = "cosine_warmup_with_min_lr"
    WARMUP_STABLE_DECAY = "warmup_stable_decay"


class OptimizerType(StrEnum):
    ADAMW_TORCH = "adamw_torch"
    ADAMW_TORCH_FUSED = "adamw_torch_fused"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_TORCH_NPU_FUSED = "adamw_torch_npu_fused"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAFACTOR = "adafactor"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    ADAMW_TORCH_4BIT = "adamw_torch_4bit"
    ADAMW_TORCH_8BIT = "adamw_torch_8bit"
    ADEMAMIX = "ademamix"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    ADAMW_BNB_8BIT = "adamw_bnb_8bit"
    ADEMAMIX_8BIT = "ademamix_8bit"
    LION_8BIT = "lion_8bit"
    LION_32BIT = "lion_32bit"
    PAGED_ADAMW_32BIT = "paged_adamw_32bit"
    PAGED_ADAMW_8BIT = "paged_adamw_8bit"
    PAGED_ADEMAMIX_32BIT = "paged_ademamix_32bit"
    PAGED_ADEMAMIX_8BIT = "paged_ademamix_8bit"
    PAGED_LION_32BIT = "paged_lion_32bit"
    PAGED_LION_8BIT = "paged_lion_8bit"
    RMSPROP = "rmsprop"
    RMSPROP_BNB = "rmsprop_bnb"
    RMSPROP_BNB_8BIT = "rmsprop_bnb_8bit"
    RMSPROP_BNB_32BIT = "rmsprop_bnb_32bit"
    GALORE_ADAMW = "galore_adamw"
    GALORE_ADAMW_8BIT = "galore_adamw_8bit"
    GALORE_ADAFACTOR = "galore_adafactor"
    GALORE_ADAMW_LAYERWISE = "galore_adamw_layerwise"
    GALORE_ADAMW_8BIT_LAYERWISE = "galore_adamw_8bit_layerwise"
    GALORE_ADAFACTOR_LAYERWISE = "galore_adafactor_layerwise"
    LOMO = "lomo"
    ADALOMO = "adalomo"
    GROKADAMW = "grokadamw"
    SCHEDULE_FREE_RADAM = "schedule_free_radam"
    SCHEDULE_FREE_ADAMW = "schedule_free_adamw"
    SCHEDULE_FREE_SGD = "schedule_free_sgd"
    APOLLO_ADAMW = "apollo_adamw"
    APOLLO_ADAMW_LAYERWISE = "apollo_adamw_layerwise"
    STABLE_ADAMW = "stable_adamw"


class SFTParameters(BaseModel):
    """LoRA and SFT trainer hyperparameters."""

    model_config = ConfigDict(frozen=True)

    rank: int = 16
    alpha_multiplier: int = 2
    use_projection_modules: bool = False
    lora_dropout: float = 0.0
    warmup_ratio: float = 0.05
    learning_rate: float = 2e-4
    optim: OptimizerType = OptimizerType.ADAMW_TORCH_8BIT
    weight_decay: float = 0.01
    lr_scheduler_type: LRSchedulerType = LRSchedulerType.LINEAR

    @property
    def training_modules(self) -> list[str]:
        training_modules: list[str] = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
        if self.use_projection_modules:
            training_modules.extend(
                [
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ]
            )
        return training_modules

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict with all enum values serialized to strings."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SFTParameters:
        """Create an SFTParameters from a plain dict (e.g. from a wandb config)."""
        return cls.model_validate(data)

    @classmethod
    def get_default_sweep_config(cls, metric_name: str = "f1", metric_goal: str = "maximize") -> dict[str, Any]:
        return {
            "method": "random",
            "metric": {"goal": metric_goal, "name": metric_name},
            "parameters": {
                "rank": {"values": [8, 16, 32]},
                "alpha_multiplier": {"values": [1, 2]},
                "use_projection_modules": {"values": [True, False]},
                "warmup_ratio": {"values": [0.05, 0.1]},
                "lr_schedular_type": {"values": ["linear", "cosine"]},
                "optim": {"values": ["adamw_bnb_8bit", "sgd"]},
                "learning_rate": {"distribution": "log_uniform_values", "min": 5e-5, "max": 2e-3},
                "lora_dropout": {"distribution": "uniform", "min": 0, "max": 0.1},
                "weight_decay": {"distribution": "uniform", "min": 0.01, "max": 0.1},
            },
        }
