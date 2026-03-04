from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable

from datasets import Dataset

from classification_trainer.configuration import DatasetInfo, InferenceInfo
from classification_trainer.protocols import MetricResult


class ClassificationResult(IntEnum):
    TRUE_POSITIVE = 0
    TRUE_NEGATIVE = 1
    FALSE_POSITIVE = 2
    FALSE_NEGATIVE = 3


def _is_positive_case(test: str, positive_case: str, search_length: int, search_from_end: bool = False) -> bool:
    if search_length >= 0:
        return positive_case in (test[-search_length:] if search_from_end else test[:search_length])
    else:
        return positive_case in test


def classify_inference(
    inference: str,
    ground_truth: str,
    dataset_info: DatasetInfo,
) -> ClassificationResult:
    positive_case = dataset_info.positive_case
    if dataset_info.case_invariant_comparison:
        inference = inference.lower()
        ground_truth = ground_truth.lower()
        positive_case = positive_case.lower()

    ground_truth_is_positive_case = _is_positive_case(
        ground_truth, positive_case, dataset_info.search_length, dataset_info.search_from_end
    )
    inference_is_positive_case = _is_positive_case(
        inference, positive_case, dataset_info.search_length, dataset_info.search_from_end
    )
    if inference_is_positive_case:
        return (
            ClassificationResult.TRUE_POSITIVE if ground_truth_is_positive_case else ClassificationResult.FALSE_POSITIVE
        )
    else:
        return (
            ClassificationResult.TRUE_NEGATIVE
            if not ground_truth_is_positive_case
            else ClassificationResult.FALSE_NEGATIVE
        )


def add_classification_result_column(
    dataset_info: DatasetInfo,
    dataset: Dataset,
) -> Dataset:
    ground_truth_column_name = dataset_info.string_labels_column_name
    inference_column_name = dataset_info.prediction_column_name
    classification_result_column_name = dataset_info.classification_result_column_name

    return dataset.map(
        lambda row: {
            classification_result_column_name: classify_inference(
                row[inference_column_name], row[ground_truth_column_name], dataset_info
            ).name
        }
    )


@dataclass
class ClassificationCounts:
    """Container for binary classification confusion matrix counts."""

    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    def update(self, result: ClassificationResult) -> None:
        if result == ClassificationResult.FALSE_NEGATIVE:
            self.false_negatives = self.false_negatives + 1
        elif result == ClassificationResult.TRUE_NEGATIVE:
            self.true_negatives = self.true_negatives + 1
        elif result == ClassificationResult.FALSE_POSITIVE:
            self.false_positives = self.false_positives + 1
        elif result == ClassificationResult.TRUE_POSITIVE:
            self.true_positives = self.true_positives + 1
        else:
            raise ValueError(f"Unknown classification result {result}")

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


def collect_classification_counts(dataset_info: DatasetInfo, dataset: Dataset) -> ClassificationCounts:
    retval: ClassificationCounts = ClassificationCounts()
    for classification in dataset[dataset_info.classification_result_column_name]:
        typed_classification: ClassificationResult = ClassificationResult[classification]
        retval.update(typed_classification)

    return retval


def generate_metrics(counts: ClassificationCounts, metric_creators: Iterable[MetricProtocol]) -> Iterator[MetricResult]:
    for metric_creator in metric_creators:
        yield metric_creator.compute_metric(counts)


class AccuracyMetric:
    """Computes accuracy: (TP + TN) / total."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        if counts.total == 0:
            return MetricResult("accuracy", 0.0)
        return MetricResult("accuracy", (counts.true_positives + counts.true_negatives) / counts.total)


class PrecisionMetric:
    """Computes precision: TP / (TP + FP). Returns 0.0 when denominator is 0."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        denominator = counts.true_positives + counts.false_positives
        if denominator == 0:
            return MetricResult("precision", 0.0)
        return MetricResult("precision", counts.true_positives / denominator)


class RecallMetric:
    """Computes recall: TP / (TP + FN). Returns 0.0 when denominator is 0."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        denominator = counts.true_positives + counts.false_negatives
        if denominator == 0:
            return MetricResult("recall", 0.0)
        return MetricResult("recall", counts.true_positives / denominator)


class F1Metric:
    """Computes F1: 2 * precision * recall / (precision + recall). Returns 0.0 when denominator is 0."""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        precision = PrecisionMetric().compute_metric(counts).metric_result
        recall = RecallMetric().compute_metric(counts).metric_result
        denominator = precision + recall
        if denominator == 0:
            return MetricResult("f1", 0.0)
        return MetricResult("f1", 2 * precision * recall / denominator)


class TotalMetric:
    """Returns the total number of seen classifications"""

    def compute_metric(self, counts: ClassificationCounts) -> MetricResult:
        return MetricResult("total_seen", counts.total)


def get_common_metric_creators() -> Iterator[MetricProtocol]:
    yield AccuracyMetric()
    yield PrecisionMetric()
    yield RecallMetric()
    yield F1Metric()
    yield TotalMetric()


_METRIC_REGISTRY: dict[str, MetricProtocol] = {
    "accuracy": AccuracyMetric(),
    "precision": PrecisionMetric(),
    "recall": RecallMetric(),
    "f1": F1Metric(),
    "total_seen": TotalMetric(),
}


def get_metrics_from_inference_info(inference_info: InferenceInfo) -> Iterator[MetricProtocol]:
    for name in inference_info.metrics:
        if name not in _METRIC_REGISTRY:
            raise ValueError(f"Unknown metric '{name}'. Valid options: {list(_METRIC_REGISTRY)}")
        yield _METRIC_REGISTRY[name]
