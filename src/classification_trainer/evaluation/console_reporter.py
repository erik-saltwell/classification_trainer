from __future__ import annotations

from collections.abc import Iterable

from classification_trainer.protocols import LoggingProtocol, MetricResult


class ConsoleMetricsReporter:
    """Reports metric results to the console via LoggingProtocol."""

    def __init__(self, logger: LoggingProtocol) -> None:
        self._logger = logger

    def report(self, results: Iterable[MetricResult]) -> None:
        """Log each metric result as a key-value table row.

        Floats are formatted to 4 decimal places before reporting.
        """
        for result in results:
            value = result.metric_result
            if isinstance(value, float):
                value = f"{value:.4f}"
            self._logger.report_table_message({result.metric_name: value})
