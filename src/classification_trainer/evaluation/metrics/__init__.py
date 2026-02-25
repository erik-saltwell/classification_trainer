from __future__ import annotations

from classification_trainer.protocols import MetricProtocol

from .accuracy import AccuracyMetric
from .f1 import F1Metric
from .precision import PrecisionMetric
from .recall import RecallMetric


def get_default_metrics() -> list[MetricProtocol]:
    """Return the default set of classification metrics."""
    return [AccuracyMetric(), PrecisionMetric(), RecallMetric(), F1Metric()]


__all__ = ["AccuracyMetric", "F1Metric", "get_default_metrics", "PrecisionMetric", "RecallMetric"]
