"""Classifier evaluation: confusion matrix, threshold sweep, ROC and PR curves.

Sensitivity and specificity alone are misleading when negatives outnumber
positives, which they do here by roughly an order of magnitude. Precision, F1
and Matthews correlation are reported alongside them, and the operating
threshold is chosen by sweeping rather than fixed by convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise

# A sequence absent from hmmsearch output scored below the reporting threshold.
# It is given a worse-than-any-reported E-value so it still counts as a negative
# prediction rather than disappearing from the denominator.
NON_HIT_EVALUE = float("inf")


@dataclass(frozen=True)
class Metrics:
    """Every metric worth quoting for a binary classifier, at one threshold."""

    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    @property
    def sensitivity(self) -> float:
        """Recall / true-positive rate."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def specificity(self) -> float:
        """True-negative rate."""
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        """Positive predictive value: of everything called positive, how much is."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def false_positive_rate(self) -> float:
        return 1.0 - self.specificity

    @property
    def f1(self) -> float:
        denominator = self.precision + self.sensitivity
        return 2 * self.precision * self.sensitivity / denominator if denominator else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.sensitivity + self.specificity) / 2

    @property
    def youden_j(self) -> float:
        return self.sensitivity + self.specificity - 1.0

    @property
    def mcc(self) -> float:
        """Matthews correlation coefficient: the imbalance-robust summary."""
        tp, fp, tn, fn = (
            self.true_positives,
            self.false_positives,
            self.true_negatives,
            self.false_negatives,
        )
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "sensitivity": round(self.sensitivity, 4),
            "specificity": round(self.specificity, 4),
            "precision": round(self.precision, 4),
            "f1": round(self.f1, 4),
            "mcc": round(self.mcc, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
        }


def evaluate_at(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    threshold: float,
    higher_is_better: bool = False,
) -> Metrics:
    """Confusion matrix at one threshold.

    Args:
        higher_is_better: ``False`` for E-values, where a *smaller* score is a
            stronger call; ``True`` for probabilities or decision scores.
    """
    if higher_is_better:
        tp = sum(1 for score in positive_scores if score >= threshold)
        fp = sum(1 for score in negative_scores if score >= threshold)
    else:
        tp = sum(1 for score in positive_scores if score <= threshold)
        fp = sum(1 for score in negative_scores if score <= threshold)

    return Metrics(
        threshold=threshold,
        true_positives=tp,
        false_positives=fp,
        true_negatives=len(negative_scores) - fp,
        false_negatives=len(positive_scores) - tp,
    )


def sweep(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    higher_is_better: bool = False,
) -> list[Metrics]:
    """Evaluate at every threshold that changes the confusion matrix."""
    finite = sorted(
        {score for score in [*positive_scores, *negative_scores] if math.isfinite(score)}
    )
    if higher_is_better:
        # A threshold above every score admits nothing, putting the sweep's
        # first point at the (0, 0) corner of ROC space.
        ceiling = (finite[-1] + 1.0) if finite else 1.0
        thresholds = [ceiling, *reversed(finite)]
    else:
        # E-values are non-negative, so a negative threshold admits nothing.
        # Dividing the smallest score instead would fail whenever hmmsearch
        # reports an E-value of exactly 0 for a very strong hit.
        thresholds = [-1.0, *finite]

    return [
        evaluate_at(positive_scores, negative_scores, t, higher_is_better) for t in thresholds
    ]


def roc_auc(curve: Sequence[Metrics]) -> float:
    """Area under the ROC curve, by the trapezoid rule."""
    points = sorted({(metrics.false_positive_rate, metrics.sensitivity) for metrics in curve})
    points = [(0.0, 0.0), *points, (1.0, 1.0)]

    area = 0.0
    for (x0, y0), (x1, y1) in pairwise(points):
        area += (x1 - x0) * (y0 + y1) / 2
    return round(area, 4)


def average_precision(curve: Sequence[Metrics]) -> float:
    """Area under the precision-recall curve."""
    points = sorted({(metrics.sensitivity, metrics.precision) for metrics in curve})
    area = 0.0
    previous_recall = 0.0
    for recall, precision in points:
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return round(area, 4)


def best_by(curve: Sequence[Metrics], criterion: str = "mcc") -> Metrics:
    """The threshold maximising a chosen criterion."""
    if not curve:
        raise ValueError("empty sweep")
    return max(curve, key=lambda metrics: getattr(metrics, criterion))


def scores_for(identifiers: Sequence[str], hits: dict[str, float]) -> list[float]:
    """Look up each sequence's E-value, defaulting to a non-hit."""
    return [hits.get(identifier, NON_HIT_EVALUE) for identifier in identifiers]
