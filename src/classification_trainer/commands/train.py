from contextlib import nullcontext
from enum import IntEnum

import typer
from attr import dataclass
from datasets import Dataset

from classification_trainer.configuration import DatasetInfo, TrainingInfo
from classification_trainer.helpers.evaluation_helper import F1Metric
from classification_trainer.helpers.reporting_helper import (
    CompositeMetricsReporter,
    LoggerMetricsReporter,
    WandBMetricsReporter,
)
from classification_trainer.helpers.wandb_helper import initialize_wandb
from classification_trainer.protocols import CommandProtocol, LoggingProtocol
from classification_trainer.protocols.metric_reporting_protocol import MetricsReportingProtocol
from classification_trainer.protocols.metric_result import MetricResult

from .training_runner import TrainingRunner


@dataclass
class DatasetSplits:
    training_dataset: Dataset
    test_dataset: Dataset
    validation_dataset: Dataset


class MetricsTrainingSteps(IntEnum):
    PRE_TRAINING = 0


@dataclass
class TrainCommand(CommandProtocol):
    training_info: TrainingInfo
    dataset_info: DatasetInfo
    pretokenize: bool

    def log_comparison_report(
        self, pre_run_results: list[MetricResult], post_run_results: list[MetricResult], logger: LoggingProtocol
    ) -> None:
        pre_dict = {r.metric_name: r.metric_result for r in pre_run_results}
        post_dict = {r.metric_name: r.metric_result for r in post_run_results}
        if pre_dict.keys() != post_dict.keys():
            raise ValueError(
                f"Metric sets differ between pre and post results: "
                f"pre={sorted(pre_dict.keys())}, post={sorted(post_dict.keys())}"
            )
        logger.report_multicolumn_table(
            headers=["metric-name", "pre-result", "post-result"],
            rows=[[name, str(pre_dict[name]), str(post_dict[name])] for name in pre_dict],
        )

    def create_metrics_reporter(self, wandb_enabled: bool, logger: LoggingProtocol) -> MetricsReportingProtocol:
        reporters: list[MetricsReportingProtocol] = [LoggerMetricsReporter(logger)]
        if wandb_enabled:
            reporters.append(WandBMetricsReporter())
        reporter = CompositeMetricsReporter(reporters)
        return reporter

    def execute(self, logger: LoggingProtocol) -> None:
        try:
            wandb_enabled = self.training_info.wandb_config is not None
            ctx = initialize_wandb(self.training_info) if wandb_enabled else nullcontext()
            with ctx:
                runner: TrainingRunner = TrainingRunner(self.training_info, self.dataset_info, self.pretokenize)
                logger.report_message("[blue]Prepare Data[/blue]")
                runner.prepare_data(logger)

                logger.report_message("[blue]Loading base model...[/blue]")
                runner.load_model(logger)

                reporter: MetricsReportingProtocol = self.create_metrics_reporter(wandb_enabled, logger)

                logger.report_message("[blue]Pre Training Assessment...[/blue]")
                pre_run_results: list[MetricResult] = runner.evaluate_model(logger, F1Metric())
                reporter.report(pre_run_results, MetricsTrainingSteps.PRE_TRAINING)

                logger.report_message("[blue]Training...[/blue]")
                final_step = runner.train_model(logger)

                logger.report_message("[blue]Post-Run Assessment...[/blue]")
                post_run_results: list[MetricResult] = runner.evaluate_model(logger, F1Metric())
                reporter.report(post_run_results, final_step)

                logger.add_break()
                self.log_comparison_report(pre_run_results, post_run_results, logger)

                if (
                    self.training_info.publishing_info is not None
                    and self.training_info.publishing_info.any_publish_enabled
                ):
                    runner.save_model(pre_run_results, post_run_results, logger)
        except Exception as e:
            logger.report_exception("Error Analyzing Dataset", e)
            raise typer.Exit(code=1) from e
