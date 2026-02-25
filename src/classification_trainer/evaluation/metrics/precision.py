from __future__ import annotations

from classification_trainer.protocols import ClassificationCounts, MetricResult


class PrecisionMetric:
    """Computes precision: TP / (TP + FP). Returns 0.0 when denominator is 0."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        denominator = counts.true_positives + counts.false_positives
        if denominator == 0:
            return MetricResult("precision", 0.0)
        return MetricResult("precision", counts.true_positives / denominator)
