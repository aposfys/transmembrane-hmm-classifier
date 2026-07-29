"""Build and evaluate profile HMMs for single-pass type I transmembrane proteins."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO

from . import data, evaluate, pipeline, regions

MODELS = ("full_length", "tm_region")


@dataclass
class Dataset:
    """Everything the model builder and evaluator need."""

    train_fasta: Path
    positive_test: list[str]
    negatives: dict[str, list[str]]
    negative_fasta: Path
    positive_fasta: Path

    @property
    def all_negatives(self) -> list[str]:
        return [name for names in self.negatives.values() for name in names]


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
    print(f"Train {len(split.train)} / test {len(split.test)} "
          f"(sums to {len(split.train) + len(split.test)} = input)")

    train_ids = list(split.train)
    if args.max_train and len(train_ids) > args.max_train:
        train_ids = sorted(
            random.Random(args.seed).sample(train_ids, args.max_train)
        )
        print(f"Capped training set at {args.max_train} sequences "
              "(both models use this same set)")

    train_fasta = pipeline.subset_fasta(
        clustered, set(train_ids), args.data_dir / "train.fasta"
    )

    negatives: dict[str, list[str]] = {}
    negative_records = []
    for name in data.NEGATIVE_CLASSES:
        proteins = data.parse_tsv(
            data.fetch_class(name, args.data_dir / f"{name}.tsv")
        )
        fasta = data.write_fasta(proteins, args.data_dir / f"{name}.fasta")
        reduced = pipeline.cluster(
            fasta,
            args.data_dir / f"{name}_nr{int(args.identity * 100)}.fasta",
            identity=args.identity,
        )
        # Sample a fixed number per class so no single decoy class dominates.
        ids = pipeline.fasta_identifiers(reduced)
        rng = random.Random(args.seed + len(name))
        sampled = sorted(rng.sample(ids, min(len(split.test), len(ids))))
        negatives[name] = sampled
        negative_records.extend(
            record
            for record in SeqIO.parse(reduced, "fasta")
            if record.id in set(sampled)
        )
        print(f"Negative class {name:9s}: {len(ids):5d} non-redundant, "
              f"{len(sampled)} sampled")

    negative_fasta = args.data_dir / "negatives.fasta"
    SeqIO.write(negative_records, negative_fasta, "fasta")

    test_positive_fasta = pipeline.subset_fasta(
        clustered, set(split.test), args.data_dir / "test_positive.fasta"
    )

    return Dataset(
        train_fasta=train_fasta,
        positive_test=list(split.test),
        negatives=negatives,
        negative_fasta=negative_fasta,
        positive_fasta=test_positive_fasta,
    )


def training_fasta_for(model: str, train_fasta: Path, args) -> Path:
    """Full-length training sequences, or just their TM regions."""
    if model == "full_length":
        return train_fasta

    proteins = data.single_tm_only(
        data.parse_tsv(args.data_dir / "type_i.tsv")
    )
    wanted = {record.id.split("|")[1] for record in SeqIO.parse(train_fasta, "fasta")}
    selected = [protein for protein in proteins if protein.accession in wanted]
    extracted = regions.extract_regions(selected, padding=args.padding)
    print(f"Extracted {len(extracted)} TM regions "
          f"(+/- {args.padding} residues of flank) from {len(selected)} sequences")
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
    print(f"At the conventional E <= 0.05 cutoff: "
          f"sens {conventional.sensitivity:.3f}, spec {conventional.specificity:.3f}, "
          f"precision {conventional.precision:.3f}, MCC {conventional.mcc:.3f}")
    print(f"At the MCC-optimal cutoff E <= {optimal.threshold:.2e}: "
          f"sens {optimal.sensitivity:.3f}, spec {optimal.specificity:.3f}, "
          f"precision {optimal.precision:.3f}, MCC {optimal.mcc:.3f}")

    per_class = {}
    for name, identifiers in dataset.negatives.items():
        class_scores = evaluate.scores_for(identifiers, negative_hits)
        false_positives = sum(
            1 for score in class_scores if score <= optimal.threshold
        )
        per_class[name] = {
            "sequences": len(identifiers),
            "false_positives": false_positives,
            "false_positive_rate": round(false_positives / len(identifiers), 4)
            if identifiers
            else 0.0,
        }
        print(f"    {name:9s}: {false_positives}/{len(identifiers)} false positives "
              f"({per_class[name]['false_positive_rate']:.1%})")

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
        print(f"    fold {index}: sens {metrics.sensitivity:.3f} "
              f"spec {metrics.specificity:.3f} MCC {metrics.mcc:.3f}")

    def summarise(key: str) -> dict[str, float]:
        values = [score[key] for score in scores]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return {"mean": round(mean, 4), "sd": round(variance**0.5, 4)}

    summary = {key: summarise(key) for key in ("sensitivity", "specificity", "precision", "mcc")}
    print(f"    {args.folds}-fold mean MCC {summary['mcc']['mean']:.3f} "
          f"+/- {summary['mcc']['sd']:.3f}")
    return {"folds": scores, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    findings: dict[str, object] = {
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
        hmm = pipeline.build_hmm(
            alignment, args.results_dir / f"{model}.hmm", name=model
        )
        findings[model] = evaluate_model(model, hmm, dataset, args)

        if args.folds and model in args.cv_models:
            print(f"  {args.folds}-fold cross-validation:")
            findings[model]["cross_validation"] = cross_validate(model, dataset, args)

    summary_path = args.results_dir / "findings.json"
    summary_path.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")

    if not args.no_figures:
        from .plots import plot_confusion, plot_curves

        plot_curves(findings, args.results_dir / "roc_pr_curves.png")
        plot_confusion(findings, args.results_dir / "confusion_matrices.png")

    print(f"\nAll results written to {args.results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
