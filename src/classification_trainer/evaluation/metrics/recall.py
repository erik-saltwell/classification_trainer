from __future__ import annotations

from classification_trainer.protocols import ClassificationCounts, MetricResult


class RecallMetric:
    """Computes recall: TP / (TP + FN). Returns 0.0 when denominator is 0."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        denominator = counts.true_positives + counts.false_negatives
        if denominator == 0:
            return MetricResult("recall", 0.0)
        return MetricResult("recall", counts.true_positives / denominator)
