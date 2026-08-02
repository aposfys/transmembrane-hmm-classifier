"""Benchmark of protein language model heads against the profile HMM.

Both families of model are scored on exactly the same test sequences, so their
numbers are directly comparable. They are *not* trained the same way, and that
asymmetry is deliberate rather than an oversight:

* A profile HMM is generative. It is built from an alignment of positives only
  and never sees a negative example.
* A linear or MLP head is discriminative. It requires labelled negatives.

The negatives used to train the heads are drawn from the pool the shared test
set did not consume, so no test sequence is ever trained on. A decoy class that
the test set exhausts contributes no training examples at all, which makes its
error rate a measure of generalisation to an unseen class.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

from Bio import SeqIO

from . import evaluate, heads, plm


def _records(
    fasta: Path, limit: int | None = None, seed: int = heads.SEED
) -> list[tuple[str, str]]:
    records = [(record.id, str(record.seq)) for record in SeqIO.parse(fasta, "fasta")]
    if limit and len(records) > limit:
        # Embedding is the dominant cost, and a linear probe saturates long
        # before the full pool is used. Sample deterministically.
        records = sorted(random.Random(seed).sample(records, limit))
    return records


def embed_all(
    dataset,
    cache_dir: Path,
    model: str = plm.DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 8,
    max_train: int | None = None,
    half_precision: bool = True,
) -> dict[str, plm.EmbeddingSet]:
    """Embed every sequence split once and reuse it across heads and folds.

    Test splits are never subsampled: they are the shared basis of comparison
    with the profile HMM. Only the head-training pools are capped.
    """
    sources = {
        "train_positive": (dataset.full_train_fasta, max_train),
        "test_positive": (dataset.positive_fasta, None),
        "test_negative": (dataset.negative_fasta, None),
    }
    if dataset.train_negative_fasta is not None:
        sources["train_negative"] = (dataset.train_negative_fasta, max_train)

    embeddings: dict[str, plm.EmbeddingSet] = {}
    for name, (fasta, limit) in sources.items():
        records = _records(fasta, limit=limit)
        print(f"  embedding {name}: {len(records)} sequences")
        embeddings[name] = plm.embed_cached(
            records,
            cache_dir,
            name,
            model=model,
            device=device,
            batch_size=batch_size,
            half_precision=half_precision,
        )
    return embeddings


def evaluate_head(
    head_name: str,
    embeddings: dict[str, plm.EmbeddingSet],
    dataset,
    seed: int = heads.SEED,
) -> dict:
    """Train one head and score it on the shared test set."""
    train_positive = embeddings["train_positive"].vectors
    train_negative = embeddings["train_negative"].vectors

    head = heads.train(head_name, train_positive, train_negative, seed=seed)

    positive_scores = head.score(
        embeddings["test_positive"].subset(dataset.positive_test)
    ).tolist()
    negative_scores = head.score(
        embeddings["test_negative"].subset(dataset.all_negatives)
    ).tolist()

    curve = evaluate.sweep(positive_scores, negative_scores, higher_is_better=True)
    auc = evaluate.roc_auc(curve)
    average_precision = evaluate.average_precision(curve)
    default = evaluate.evaluate_at(
        positive_scores, negative_scores, 0.5, higher_is_better=True
    )
    optimal = evaluate.best_by(curve, "mcc")

    print(f"\n--- {head_name} ---")
    print(f"ROC AUC {auc:.3f} | average precision {average_precision:.3f}")
    print(
        f"At the default p >= 0.5 cutoff: "
        f"sens {default.sensitivity:.3f}, spec {default.specificity:.3f}, "
        f"precision {default.precision:.3f}, MCC {default.mcc:.3f}"
    )
    print(
        f"At the MCC-optimal cutoff p >= {optimal.threshold:.3f}: "
        f"sens {optimal.sensitivity:.3f}, spec {optimal.specificity:.3f}, "
        f"precision {optimal.precision:.3f}, MCC {optimal.mcc:.3f}"
    )

    per_class = {}
    offset = 0
    for name, identifiers in dataset.negatives.items():
        class_scores = negative_scores[offset : offset + len(identifiers)]
        offset += len(identifiers)
        false_positives = sum(1 for s in class_scores if s >= optimal.threshold)
        held_out = not dataset.train_negatives.get(name)
        per_class[name] = {
            "sequences": len(identifiers),
            "false_positives": false_positives,
            "false_positive_rate": round(false_positives / len(identifiers), 4)
            if identifiers
            else 0.0,
            "held_out_from_training": held_out,
        }
        marker = "  (never trained on)" if held_out else ""
        print(
            f"    {name:9s}: {false_positives}/{len(identifiers)} false positives "
            f"({per_class[name]['false_positive_rate']:.1%}){marker}"
        )

    separability = heads.embedding_separability(train_positive, train_negative)
    print(
        f"    embedding separation ratio {separability['separation_ratio']:.3f} "
        f"(centroid distance / within-class spread)"
    )

    return {
        "head": head_name,
        "embedding_model": embeddings["train_positive"].model,
        "embedding_dimension": embeddings["train_positive"].dimension,
        "training_positives": len(train_positive),
        "training_negatives": len(train_negative),
        "roc_auc": auc,
        "average_precision": average_precision,
        "at_default_cutoff": default.as_dict(),
        "at_optimal_cutoff": optimal.as_dict(),
        "false_positives_by_class": per_class,
        "embedding_separability": separability,
        "curve": [
            {
                "threshold": m.threshold,
                "fpr": round(m.false_positive_rate, 4),
                "tpr": round(m.sensitivity, 4),
                "precision": round(m.precision, 4),
            }
            for m in curve
        ],
    }


def out_of_fold_check(
    head_name: str, embeddings: dict[str, plm.EmbeddingSet], folds: int = 5
) -> dict:
    """Cross-validate on the training data alone, without touching the test set.

    A large gap between this and the held-out test score is the signature of a
    head that has memorised its training pool.
    """
    positive_scores, negative_scores = heads.cross_validated_scores(
        head_name,
        embeddings["train_positive"].vectors,
        embeddings["train_negative"].vectors,
        folds=folds,
    )
    curve = evaluate.sweep(positive_scores, negative_scores, higher_is_better=True)
    best = evaluate.best_by(curve, "mcc")
    result = {
        "folds": folds,
        "roc_auc": evaluate.roc_auc(curve),
        "mcc": round(best.mcc, 4),
    }
    print(
        f"    out-of-fold on training data: AUC {result['roc_auc']:.3f}, "
        f"MCC {result['mcc']:.3f}"
    )
    return result


def run(
    dataset,
    head_names: Sequence[str],
    cache_dir: Path,
    model: str = plm.DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 8,
    folds: int = 5,
    max_train: int | None = None,
    half_precision: bool = True,
) -> dict[str, dict]:
    """Embed once, then train and evaluate each requested head."""
    if dataset.train_negative_fasta is None:
        raise RuntimeError(
            "No negative sequences are available for head training. Lower "
            "--test-fraction so the decoy pools are not fully consumed by the "
            "shared test set."
        )

    print(f"\n=== Protein language model benchmark (ESM-2 {model}) ===")
    embeddings = embed_all(
        dataset,
        cache_dir,
        model=model,
        device=device,
        batch_size=batch_size,
        max_train=max_train,
        half_precision=half_precision,
    )

    findings = {}
    for head_name in head_names:
        findings[f"esm_{head_name}"] = evaluate_head(head_name, embeddings, dataset)
        if folds:
            findings[f"esm_{head_name}"]["out_of_fold"] = out_of_fold_check(
                head_name, embeddings, folds=folds
            )
    return findings
