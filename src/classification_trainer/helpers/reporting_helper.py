from collections.abc import Iterable
from dataclasses import dataclass

import wandb
from classification_trainer.protocols import MetricResult, MetricsReportingProtocol
from classification_trainer.protocols.logging_protocol import LoggingProtocol


@dataclass
class LoggerMetricsReporter(MetricsReportingProtocol):
    logger: LoggingProtocol

    def report(self, results: Iterable[MetricResult], step: int) -> None:
        result_list = list(results)
        self.logger.report_message(f"Metrics for step {step}")
        self.logger.report_multicolumn_table(
            headers=["metric", "result"],
            rows=[[r.metric_name, str(r.metric_result)] for r in result_list],
        )


def metrics_list_to_dict(results: Iterable[MetricResult], prefix: str = "") -> dict[str, float]:
    return {f"{prefix}{r.metric_name}": r.metric_result for r in results}


@dataclass
class WandBMetricsReporter(MetricsReportingProtocol):
    def report(self, results: Iterable[MetricResult], step: int) -> None:
        wandb.log(metrics_list_to_dict(results), step=step)


@dataclass
class CompositeMetricsReporter(MetricsReportingProtocol):
    reporters: list[MetricsReportingProtocol]

    def report(self, results: Iterable[MetricResult], step: int) -> None:
        result_list = list(results)
        for reporter in self.reporters:
            reporter.report(result_list, step)
