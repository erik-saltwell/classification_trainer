from __future__ import annotations

from classification_trainer.protocols import ClassificationCounts


def count_confusion_matrix(
    predictions: list[str],
    ground_truths: list[str],
    positive_class: str,
) -> ClassificationCounts:
    """Count TP/TN/FP/FN for a binary classification task.

    Comparison is case-sensitive exact match. Cleaning should have already
    normalised case before calling this function.

    Args:
        predictions: Cleaned model predictions.
        ground_truths: Ground-truth labels from the dataset.
        positive_class: The label that counts as "positive".

    Returns:
        A ClassificationCounts with tallied TP/TN/FP/FN.
    """
    counts = ClassificationCounts()
    for pred, truth in zip(predictions, ground_truths, strict=False):
        if truth == positive_class:
            if pred == positive_class:
                counts.true_positives += 1
            else:
                counts.false_negatives += 1
        else:
            if pred == positive_class:
                counts.false_positives += 1
            else:
                counts.true_negatives += 1
    return counts
