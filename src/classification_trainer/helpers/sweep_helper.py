from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from classification_trainer.configuration.inference_info import InferenceInfo
from classification_trainer.configuration.sft_parameters import SFTParameters
from classification_trainer.configuration.training_info import TrainingInfo
from classification_trainer.protocols.metric_result import MetricResult


def build_sweep_config(inference_info: InferenceInfo) -> dict[str, Any]:
    """Build the wandb sweep configuration dict from the inference profile."""
    return SFTParameters.get_default_sweep_config(
        metric_name=inference_info.sweep_metric,
        metric_goal=inference_info.sweep_metric_goal,
    )


def apply_trial_sft_parameters(
    training_info: TrainingInfo,
    trial_config: Mapping[str, Any],
) -> TrainingInfo:
    """Return a copy of training_info with sft_parameters replaced by the trial's values."""
    sft_params = SFTParameters.from_dict(dict(trial_config))
    return training_info.model_copy(update={"sft_parameters": sft_params})


def extract_target_metric(results: list[MetricResult], metric_name: str) -> float:
    """Return the numeric value of the named metric from a results list.

    Raises:
        ValueError: If no result with the given metric_name is found.
    """
    for result in results:
        if result.metric_name == metric_name:
            return float(result.metric_result)
    raise ValueError(f"Metric '{metric_name}' not found in results. Available: {[r.metric_name for r in results]}")
