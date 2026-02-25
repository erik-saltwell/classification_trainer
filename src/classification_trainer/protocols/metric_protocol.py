from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .metric_result import MetricResult


@dataclass
class ClassificationCounts:
    """Container for binary classification confusion matrix counts."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def total(self) -> int:
        """Total number of predictions."""
        return self.true_positives + self.true_negatives + self.false_positives + self.false_negatives


@runtime_checkable
class MetricProtocol(Protocol):
    """Protocol for computing a single classification metric from confusion matrix counts."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        """Compute the metric and return a named result."""
        ...
