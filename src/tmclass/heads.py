"""Discriminative classifier heads over protein language model embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

SEED = 20250101


@dataclass(frozen=True)
class Head:
    """A trained classifier plus the name it is reported under."""

    name: str
    model: Any  # sklearn Pipeline; imported lazily so scikit-learn stays optional

    def score(self, vectors: np.ndarray) -> np.ndarray:
        """Probability that each row is a single-pass type I protein."""
        return self.model.predict_proba(vectors)[:, 1]


def logistic_regression(seed: int = SEED) -> Pipeline:
    """Linear probe. The honest baseline for embedding quality.

    If a linear model on frozen embeddings already separates the classes, the
    representation itself carries the signal and a deeper head adds little.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    C=1.0,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def multilayer_perceptron(seed: int = SEED) -> Pipeline:
    """A small non-linear head, to test whether the boundary is non-linear."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(256, 64),
                    activation="relu",
                    alpha=1e-3,
                    max_iter=600,
                    early_stopping=True,
                    n_iter_no_change=25,
                    random_state=seed,
                ),
            ),
        ]
    )


HEADS = {
    "logreg": logistic_regression,
    "mlp": multilayer_perceptron,
}


def train(
    name: str,
    positive_vectors: np.ndarray,
    negative_vectors: np.ndarray,
    seed: int = SEED,
) -> Head:
    """Fit a head on labelled embeddings.

    Class weights are balanced because the negative pool is larger than the
    positive one, and an unweighted fit would simply learn to predict the
    majority class.
    """
    if name not in HEADS:
        raise KeyError(f"Unknown head {name!r}; expected one of {sorted(HEADS)}")

    features = np.vstack([positive_vectors, negative_vectors])
    labels = np.concatenate(
        [np.ones(len(positive_vectors), dtype=int), np.zeros(len(negative_vectors), dtype=int)]
    )

    model = HEADS[name](seed)
    model.fit(features, labels)
    return Head(name=name, model=model)


def cross_validated_scores(
    name: str,
    positive_vectors: np.ndarray,
    negative_vectors: np.ndarray,
    folds: int = 5,
    seed: int = SEED,
) -> tuple[list[float], list[float]]:
    """Out-of-fold probabilities for the training data.

    Used to check the head is not simply memorising, without touching the test
    set. Returns ``(positive_scores, negative_scores)``.
    """
    from sklearn.model_selection import StratifiedKFold

    features = np.vstack([positive_vectors, negative_vectors])
    labels = np.concatenate(
        [np.ones(len(positive_vectors), dtype=int), np.zeros(len(negative_vectors), dtype=int)]
    )

    out_of_fold = np.zeros(len(labels), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_index, test_index in splitter.split(features, labels):
        model = HEADS[name](seed)
        model.fit(features[train_index], labels[train_index])
        out_of_fold[test_index] = model.predict_proba(features[test_index])[:, 1]

    return (
        out_of_fold[labels == 1].tolist(),
        out_of_fold[labels == 0].tolist(),
    )


def embedding_separability(
    positive_vectors: np.ndarray, negative_vectors: np.ndarray
) -> dict[str, float]:
    """How far apart the two classes sit before any classifier is fitted.

    Reported so the result can be attributed to the representation rather than
    to the head. Distances use the class centroids in embedding space.
    """
    positive_centre = positive_vectors.mean(axis=0)
    negative_centre = negative_vectors.mean(axis=0)

    separation = float(np.linalg.norm(positive_centre - negative_centre))
    spread = float(
        np.mean(
            [
                np.linalg.norm(positive_vectors - positive_centre, axis=1).mean(),
                np.linalg.norm(negative_vectors - negative_centre, axis=1).mean(),
            ]
        )
    )
    cosine = float(
        positive_centre
        @ negative_centre
        / (np.linalg.norm(positive_centre) * np.linalg.norm(negative_centre))
    )

    return {
        "centroid_distance": round(separation, 4),
        "mean_within_class_spread": round(spread, 4),
        "separation_ratio": round(separation / spread, 4) if spread else 0.0,
        "centroid_cosine_similarity": round(cosine, 4),
    }
