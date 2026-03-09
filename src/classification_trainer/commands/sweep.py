from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import typer

import wandb
from classification_trainer.commands.training_runner import TrainingRunner
from classification_trainer.configuration import DatasetInfo, TrainingInfo
from classification_trainer.configuration.sweep_info import SweepInfo
from classification_trainer.helpers.evaluation_helper import (
    _METRIC_REGISTRY,
    F1Metric,
)
from classification_trainer.helpers.reporting_helper import (
    CompositeMetricsReporter,
    LoggerMetricsReporter,
    WandBMetricsReporter,
)
from classification_trainer.helpers.wandb_helper import WandBJobType, initialize_wandb
from classification_trainer.protocols import CommandProtocol, LoggingProtocol
from classification_trainer.protocols.logging_protocol import NullLogger
from classification_trainer.protocols.metric_reporting_protocol import MetricsReportingProtocol
from classification_trainer.protocols.metric_result import MetricResult
from classification_trainer.utils.common_paths import CommonPaths


@dataclass
class SweepCommand(CommandProtocol):
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    logger: LoggingProtocol = NullLogger()
    runner: TrainingRunner = field(init=False)
    reporter: MetricsReportingProtocol = field(init=False)

    def __post_init__(self) -> None:
        self.runner = TrainingRunner(self.training_info, self.dataset_info)
        self.reporter = CompositeMetricsReporter([])

    def validate_parameters(self) -> None:
        # --- Startup validation ---
        if not self.training_info.has_wandb:
            self.logger.report_message(
                "[red]Error:[/red] wandb_project_name must be set in training_info to use the sweep command."
            )
            raise typer.Exit(code=1)

        if self.training_info.sweep_config is None:
            self.logger.report_message(
                "[red]Error:[/red] sweep_config must be set in training_info to use the sweep command."
            )
            raise typer.Exit(code=1)

        assert self.training_info.sweep_config is not None
        sweep_metric = self.training_info.sweep_config.metric
        sweep_metric_goal = self.training_info.sweep_config.metric_goal

        if sweep_metric not in _METRIC_REGISTRY:
            self.logger.report_message(
                f"[red]Error:[/red] sweep metric '{sweep_metric}' is not a valid metric. "
                f"Valid options: {sorted(_METRIC_REGISTRY.keys())}"
            )
            raise typer.Exit(code=1)

        if sweep_metric_goal not in ("maximize", "minimize"):
            self.logger.report_message(
                f"[red]Error:[/red] sweep metric_goal '{sweep_metric_goal}' is invalid. "
                "Must be 'maximize' or 'minimize'."
            )
            raise typer.Exit(code=1)

    def prepare_data(self) -> None:
        self.runner.prepare_data(self.logger)

    def prepare_metrics_reporter(self) -> None:
        reporters: list[MetricsReportingProtocol] = [LoggerMetricsReporter(self.logger)]
        reporters.append(WandBMetricsReporter())
        self.reporter = CompositeMetricsReporter(reporters)

    def generate_sweep_parameters(self) -> dict[str, Any]:
        assert self.training_info.sweep_config is not None
        return self.training_info.sweep_config.to_wandb_sweep_config(self.training_info.sft_parameters)

    def initialize_sweep(self, project: str, sweep_config: SweepInfo, sweep_parameters: dict[str, Any]) -> str:
        return wandb.sweep(sweep=sweep_parameters, project=project)

    def run_sweeps(self, sweep_id: str) -> None:
        assert self.training_info.sweep_config is not None
        max_run_count: int = self.training_info.sweep_config.run_cap
        current_run_count: int = 1
        self.logger.report_message(
            "[blue]Sweep optimising for: "
            f"{self.training_info.sweep_config.metric} ("
            f"{self.training_info.sweep_config.metric_goal})[/blue]"
        )

        def _run_trial() -> None:
            nonlocal current_run_count
            nonlocal max_run_count
            self.logger.report_message(
                f"[blue]***************** Sweep {current_run_count} of {max_run_count} *****************[/blue]"
            )
            self.run_single_trial()
            current_run_count = current_run_count + 1

        # --- Launch agent ---
        wandb.agent(sweep_id=sweep_id, function=_run_trial)
        return

    def run_single_trial(self) -> None:
        ctx = initialize_wandb(self.training_info, self.dataset_info, WandBJobType.SWEEP)
        with ctx:
            # self.training_info = apply_trial_sft_parameters(self.training_info, wandb.config))
            self.runner.load_model(self.logger)
            current_step: int = self.runner.train_model(self.logger)
            results: list[MetricResult] = self.runner.evaluate_model(self.logger, F1Metric())
            self.reporter.report(results, current_step + 1)

    def execute(self, logger: LoggingProtocol) -> None:
        self.logger = logger
        CommonPaths.get().clear_cache_model_directories(self.training_info.model_name)
        try:
            self.validate_parameters()
            assert self.training_info.has_wandb and self.training_info.sweep_config is not None
            assert self.training_info.wandb_project_name is not None
            self.prepare_metrics_reporter()
            self.prepare_data()
            parameters: dict[str, Any] = self.generate_sweep_parameters()

            sweep_id: str = self.initialize_sweep(
                self.training_info.wandb_project_name,
                self.training_info.sweep_config,
                parameters,
            )
            self.run_sweeps(sweep_id)

        except Exception as e:
            logger.report_exception("Error Performing Sweep", e)
            raise typer.Exit(code=1) from e
