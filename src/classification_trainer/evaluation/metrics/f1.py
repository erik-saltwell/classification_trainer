from __future__ import annotations

from classification_trainer.protocols import ClassificationCounts, MetricResult

from .precision import PrecisionMetric
from .recall import RecallMetric


class F1Metric:
    """Computes F1: 2 * precision * recall / (precision + recall). Returns 0.0 when denominator is 0."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        precision = PrecisionMetric().compute_metric(counts).metric_result
        recall = RecallMetric().compute_metric(counts).metric_result
        denominator = precision + recall
        if denominator == 0:
            return MetricResult("f1", 0.0)
        return MetricResult("f1", 2 * precision * recall / denominator)
