from __future__ import annotations

from classification_trainer.protocols import ClassificationCounts, MetricResult


class AccuracyMetric:
    """Computes accuracy: (TP + TN) / total."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        if counts.total == 0:
            return MetricResult("accuracy", 0.0)
        return MetricResult("accuracy", (counts.true_positives + counts.true_negatives) / counts.total)
