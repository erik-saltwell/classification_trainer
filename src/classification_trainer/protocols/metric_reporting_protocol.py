from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .metric_result import MetricResult


class MetricsReportingProtocol(Protocol):
    def report(self, results: Iterable[MetricResult], step: int) -> None: ...
