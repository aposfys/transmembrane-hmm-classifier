"""Build and evaluate profile HMMs for single-pass type I transmembrane proteins."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Bio import SeqIO

from . import data, evaluate, heads, pipeline, plm, regions

MODELS = ("full_length", "tm_region")


@dataclass
class Dataset:
    """Everything the model builders and the evaluator need.

    ``negatives`` is the shared test set: every model, generative or
    discriminative, is scored on exactly these sequences. ``train_negatives``
    is a disjoint pool that only the discriminative heads use, since a profile
    HMM is trained on positives alone and never sees a negative.
    """

    train_fasta: Path
    positive_test: list[str]
    negatives: dict[str, list[str]]
    negative_fasta: Path
    positive_fasta: Path
    train_negatives: dict[str, list[str]]
    train_negative_fasta: Path | None
    full_train_fasta: Path

    @property
    def all_negatives(self) -> list[str]:
        return [name for names in self.negatives.values() for name in names]

    @property
    def all_train_negatives(self) -> list[str]:
        return [name for names in self.train_negatives.values() for name in names]

    @property
    def held_out_decoy_classes(self) -> list[str]:
        """Decoy classes with no training examples, so results are generalisation."""
        return sorted(name for name, ids in self.train_negatives.items() if not ids)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tmclass",
        description=(
            "Build profile HMMs that recognise single-pass type I transmembrane "
            "proteins, and evaluate them against type II, GPCR and globular decoys."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Which model variants to build. Default: both.",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=pipeline.IDENTITY_THRESHOLD,
        help="CD-HIT sequence-identity ceiling. Default: 0.4",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=regions.DEFAULT_PADDING,
        help="Residues of flanking sequence kept either side of each TM segment.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--max-train",
        type=int,
        default=150,
        help=(
            "Cap on training sequences. Both models are built from the same "
            "capped set, so they differ only in representation and the "
            "comparison stays controlled. Whole-protein alignment is quadratic "
            "in sequence length, so raising this mostly costs full_length time "
            "(150 sequences ~4 min, 400 ~30 min). 0 removes the cap. Default: 150."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Cross-validation folds. 0 disables cross-validation.",
    )
    parser.add_argument(
        "--cv-models",
        nargs="*",
        choices=MODELS,
        default=["tm_region"],
        help=(
            "Which models to cross-validate. Defaults to tm_region only: each "
            "full_length fold needs another whole-protein alignment, which "
            "dominates the runtime."
        ),
    )
    parser.add_argument("--seed", type=int, default=20250101)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-figures", action="store_true")

    plm_group = parser.add_argument_group("protein language model benchmark")
    plm_group.add_argument(
        "--heads",
        nargs="*",
        choices=sorted(heads.HEADS),
        default=["logreg", "mlp"],
        help=(
            "Discriminative heads to train over ESM-2 embeddings and benchmark "
            "against the profile HMM on the same test set. Empty disables the "
            "benchmark."
        ),
    )
    plm_group.add_argument(
        "--esm-model",
        default=plm.DEFAULT_MODEL,
        help=(
            "ESM-2 checkpoint: a size key "
            f"({', '.join(plm.ESM2_MODELS)}) or a HuggingFace model name. "
            f"Default: {plm.DEFAULT_MODEL}."
        ),
    )
    plm_group.add_argument(
        "--device",
        default="auto",
        help="torch device for embedding: auto, mps, cuda or cpu. Default: auto.",
    )
    plm_group.add_argument(
        "--embed-batch-size",
        type=int,
        default=8,
        help="Sequence windows per forward pass. Default: 8.",
    )
    plm_group.add_argument(
        "--max-train-plm",
        type=int,
        default=600,
        help=(
            "Cap on sequences used to train each head. Embedding is the dominant "
            "cost and a linear probe saturates well below the full pool. Test "
            "splits are never capped. 0 removes the cap. Default: 600."
        ),
    )
    plm_group.add_argument(
        "--full-precision",
        action="store_true",
        help="Embed in float32 instead of float16 (slower, no practical gain).",
    )
    return parser.parse_args(argv)


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args) -> Dataset:
    """Download every class, remove redundancy, and split into train and test."""
    print("=== Dataset ===")

    positives = data.single_tm_only(
        data.parse_tsv(data.fetch_class("type_i", args.data_dir / "type_i.tsv"))
    )
    print(f"Single-pass type I with exactly one annotated TM segment: {len(positives)}")

    positive_fasta = data.write_fasta(positives, args.data_dir / "type_i.fasta")
    clustered = pipeline.cluster(
        positive_fasta,
        args.data_dir / f"type_i_nr{int(args.identity * 100)}.fasta",
        identity=args.identity,
    )
    identifiers = pipeline.fasta_identifiers(clustered)
    print(f"After CD-HIT at {args.identity:.0%} identity: {len(identifiers)}")

    split = pipeline.split_identifiers(
        identifiers, test_fraction=args.test_fraction, seed=args.seed
    )
    print(
        f"Train {len(split.train)} / test {len(split.test)} "
        f"(sums to {len(split.train) + len(split.test)} = input)"
    )

    train_ids = list(split.train)
    if args.max_train and len(train_ids) > args.max_train:
        train_ids = sorted(random.Random(args.seed).sample(train_ids, args.max_train))
        print(
            f"Capped training set at {args.max_train} sequences "
            "(both models use this same set)"
        )

    train_fasta = pipeline.subset_fasta(
        clustered, set(train_ids), args.data_dir / "train.fasta"
    )
    # The discriminative heads train on every available positive, not the capped
    # subset the alignment-based models need.
    full_train_fasta = pipeline.subset_fasta(
        clustered, set(split.train), args.data_dir / "train_full.fasta"
    )

    negatives: dict[str, list[str]] = {}
    train_negatives: dict[str, list[str]] = {}
    negative_records: list = []
    train_negative_records: list = []
    for name in data.NEGATIVE_CLASSES:
        proteins = data.parse_tsv(data.fetch_class(name, args.data_dir / f"{name}.tsv"))
        fasta = data.write_fasta(proteins, args.data_dir / f"{name}.fasta")
        reduced = pipeline.cluster(
            fasta,
            args.data_dir / f"{name}_nr{int(args.identity * 100)}.fasta",
            identity=args.identity,
        )
        # Sample a fixed number per class so no single decoy class dominates,
        # but never take more than half a class: the remainder has to be able to
        # train the discriminative heads, or a class the test set exhausts would
        # be scored by a model that has never seen anything like it.
        ids = pipeline.fasta_identifiers(reduced)
        rng = random.Random(args.seed + len(name))
        quota = min(len(split.test), len(ids) // 2)
        sampled = sorted(rng.sample(ids, quota))
        negatives[name] = sampled
        negative_records.extend(
            record for record in SeqIO.parse(reduced, "fasta") if record.id in set(sampled)
        )

        # Whatever the shared test set did not consume becomes the training pool
        # for the discriminative heads. A class exhausted by the test set is
        # therefore never trained on, and its errors measure generalisation.
        remaining = sorted(set(ids) - set(sampled))
        train_negatives[name] = remaining
        train_negative_records.extend(
            record for record in SeqIO.parse(reduced, "fasta") if record.id in set(remaining)
        )
        print(
            f"Negative class {name:9s}: {len(ids):5d} non-redundant, "
            f"{len(sampled):4d} in the shared test set, "
            f"{len(remaining):4d} available for head training"
        )

    negative_fasta = args.data_dir / "negatives.fasta"
    SeqIO.write(negative_records, negative_fasta, "fasta")

    train_negative_fasta = None
    if train_negative_records:
        train_negative_fasta = args.data_dir / "train_negatives.fasta"
        SeqIO.write(train_negative_records, train_negative_fasta, "fasta")

    test_positive_fasta = pipeline.subset_fasta(
        clustered, set(split.test), args.data_dir / "test_positive.fasta"
    )

    dataset = Dataset(
        train_fasta=train_fasta,
        positive_test=list(split.test),
        negatives=negatives,
        negative_fasta=negative_fasta,
        positive_fasta=test_positive_fasta,
        train_negatives=train_negatives,
        train_negative_fasta=train_negative_fasta,
        full_train_fasta=full_train_fasta,
    )
    if dataset.held_out_decoy_classes:
        print(
            "Decoy classes fully consumed by the test set, so never trained on: "
            + ", ".join(dataset.held_out_decoy_classes)
        )
    return dataset


def training_fasta_for(model: str, train_fasta: Path, args) -> Path:
    """Full-length training sequences, or just their TM regions."""
    if model == "full_length":
        return train_fasta

    proteins = data.single_tm_only(data.parse_tsv(args.data_dir / "type_i.tsv"))
    wanted = {record.id.split("|")[1] for record in SeqIO.parse(train_fasta, "fasta")}
    selected = [protein for protein in proteins if protein.accession in wanted]
    extracted = regions.extract_regions(selected, padding=args.padding)
    print(
        f"Extracted {len(extracted)} TM regions "
        f"(+/- {args.padding} residues of flank) from {len(selected)} sequences"
    )
    return regions.write_fasta(extracted, args.data_dir / "train_tm_regions.fasta")


def evaluate_model(model: str, hmm: Path, dataset: Dataset, args) -> dict:
    """Score the positive and negative test sets and sweep the threshold."""
    positive_hits = pipeline.parse_tblout(
        pipeline.search(
            hmm,
            dataset.positive_fasta,
            args.results_dir / f"{model}_positive.tbl",
            threads=args.threads,
        )
    )
    negative_hits = pipeline.parse_tblout(
        pipeline.search(
            hmm,
            dataset.negative_fasta,
            args.results_dir / f"{model}_negative.tbl",
            threads=args.threads,
        )
    )

    positive_scores = evaluate.scores_for(dataset.positive_test, positive_hits)
    negative_scores = evaluate.scores_for(dataset.all_negatives, negative_hits)

    curve = evaluate.sweep(positive_scores, negative_scores)
    auc = evaluate.roc_auc(curve)
    ap = evaluate.average_precision(curve)
    conventional = evaluate.evaluate_at(positive_scores, negative_scores, 0.05)
    optimal = evaluate.best_by(curve, "mcc")

    print(f"\n--- {model} ---")
    print(f"ROC AUC {auc:.3f} | average precision {ap:.3f}")
    print(
        f"At the conventional E <= 0.05 cutoff: "
        f"sens {conventional.sensitivity:.3f}, spec {conventional.specificity:.3f}, "
        f"precision {conventional.precision:.3f}, MCC {conventional.mcc:.3f}"
    )
    print(
        f"At the MCC-optimal cutoff E <= {optimal.threshold:.2e}: "
        f"sens {optimal.sensitivity:.3f}, spec {optimal.specificity:.3f}, "
        f"precision {optimal.precision:.3f}, MCC {optimal.mcc:.3f}"
    )

    per_class = {}
    for name, identifiers in dataset.negatives.items():
        class_scores = evaluate.scores_for(identifiers, negative_hits)
        false_positives = sum(1 for score in class_scores if score <= optimal.threshold)
        per_class[name] = {
            "sequences": len(identifiers),
            "false_positives": false_positives,
            "false_positive_rate": round(false_positives / len(identifiers), 4)
            if identifiers
            else 0.0,
        }
        print(
            f"    {name:9s}: {false_positives}/{len(identifiers)} false positives "
            f"({per_class[name]['false_positive_rate']:.1%})"
        )

    _write_csv([m.as_dict() for m in curve], args.results_dir / f"{model}_sweep.csv")

    return {
        "roc_auc": auc,
        "average_precision": ap,
        "at_conventional_cutoff": conventional.as_dict(),
        "at_optimal_cutoff": optimal.as_dict(),
        "false_positives_by_class": per_class,
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


def cross_validate(model: str, dataset: Dataset, args) -> dict:
    """Repeat build-and-score over k folds of the positive training set."""
    identifiers = pipeline.fasta_identifiers(dataset.train_fasta)
    rng = random.Random(args.seed)
    shuffled = list(identifiers)
    rng.shuffle(shuffled)
    folds = [shuffled[index :: args.folds] for index in range(args.folds)]

    scores: list[dict[str, float]] = []
    for index, held_out in enumerate(folds):
        held_out_set = set(held_out)
        remainder = [i for i in shuffled if i not in held_out_set]
        fold_dir = args.results_dir / "cv" / model / f"fold{index}"

        fold_train = pipeline.subset_fasta(
            dataset.train_fasta, set(remainder), fold_dir / "train.fasta"
        )
        if model == "tm_region":
            proteins = data.single_tm_only(data.parse_tsv(args.data_dir / "type_i.tsv"))
            wanted = {i.split("|")[1] for i in remainder}
            fold_train = regions.write_fasta(
                regions.extract_regions(
                    [p for p in proteins if p.accession in wanted], padding=args.padding
                ),
                fold_dir / "train_tm.fasta",
            )

        hmm = pipeline.build_hmm(
            pipeline.align(fold_train, fold_dir / "train.sto", threads=args.threads),
            fold_dir / "model.hmm",
            name=f"{model}_fold{index}",
        )

        held_out_fasta = pipeline.subset_fasta(
            dataset.train_fasta, set(held_out), fold_dir / "held_out.fasta"
        )
        positive_hits = pipeline.parse_tblout(
            pipeline.search(hmm, held_out_fasta, fold_dir / "positive.tbl", args.threads)
        )
        negative_hits = pipeline.parse_tblout(
            pipeline.search(
                hmm, dataset.negative_fasta, fold_dir / "negative.tbl", args.threads
            )
        )

        metrics = evaluate.evaluate_at(
            evaluate.scores_for(held_out, positive_hits),
            evaluate.scores_for(dataset.all_negatives, negative_hits),
            threshold=0.05,
        )
        scores.append(
            {
                "fold": index,
                "sensitivity": round(metrics.sensitivity, 4),
                "specificity": round(metrics.specificity, 4),
                "precision": round(metrics.precision, 4),
                "mcc": round(metrics.mcc, 4),
            }
        )
        print(
            f"    fold {index}: sens {metrics.sensitivity:.3f} "
            f"spec {metrics.specificity:.3f} MCC {metrics.mcc:.3f}"
        )

    def summarise(key: str) -> dict[str, float]:
        values = [score[key] for score in scores]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean": round(mean, 4), "sd": round(variance**0.5, 4)}

    summary = {
        key: summarise(key) for key in ("sensitivity", "specificity", "precision", "mcc")
    }
    print(
        f"    {args.folds}-fold mean MCC {summary['mcc']['mean']:.3f} "
        f"+/- {summary['mcc']['sd']:.3f}"
    )
    return {"folds": scores, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    findings: dict[str, Any] = {
        "dataset": {
            "positive_test": len(dataset.positive_test),
            "negatives_by_class": {k: len(v) for k, v in dataset.negatives.items()},
            "train": len(pipeline.fasta_identifiers(dataset.train_fasta)),
        }
    }

    print("\n=== Models ===")
    for model in args.models:
        training = training_fasta_for(model, dataset.train_fasta, args)
        alignment = pipeline.align(
            training, args.results_dir / f"{model}.sto", threads=args.threads
        )
        hmm = pipeline.build_hmm(alignment, args.results_dir / f"{model}.hmm", name=model)
        findings[model] = evaluate_model(model, hmm, dataset, args)

        if args.folds and model in args.cv_models:
            print(f"  {args.folds}-fold cross-validation:")
            findings[model]["cross_validation"] = cross_validate(model, dataset, args)

    if args.heads:
        # Imported here so the base install does not need scikit-learn or torch.
        from . import benchmark

        findings.update(
            benchmark.run(
                dataset,
                args.heads,
                cache_dir=args.data_dir / "embeddings",
                model=args.esm_model,
                device=args.device,
                batch_size=args.embed_batch_size,
                folds=args.folds,
                max_train=args.max_train_plm or None,
                half_precision=not args.full_precision,
            )
        )

    summary_path = args.results_dir / "findings.json"
    summary_path.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")

    if not args.no_figures:
        from .plots import (
            plot_confusion,
            plot_curves,
            plot_error_by_decoy_class,
            plot_model_comparison,
        )

        figures = {
            "roc_pr_curves.png": plot_curves,
            "confusion_matrices.png": plot_confusion,
            "model_comparison.png": plot_model_comparison,
            "error_by_decoy_class.png": plot_error_by_decoy_class,
        }
        for filename, draw in figures.items():
            # findings.json is already on disk; a plotting failure must not
            # discard hours of embedding and search work.
            try:
                draw(findings, args.results_dir / filename)
            except Exception as error:
                print(f"  figure {filename} failed: {error}")

    print(f"\nAll results written to {args.results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
