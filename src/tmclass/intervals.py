"""Confidence intervals for the reported operating points.

Every headline number in this project is an estimate from one held-out test set
of 358 positives and 888 decoys, and several of the most quoted ones -- the
per-decoy-class false-positive rates -- rest on counts small enough that the
point estimate alone is misleading. A false-positive rate of 0/172 is not
evidence that the rate is zero.

Two standard interval methods are used, chosen because both are computable from
the counts this pipeline already saves, so no model is re-run and no embedding
is recomputed:

* **Wilson score intervals** for proportions (sensitivity, specificity,
  precision, per-class false-positive rate). Wilson is used rather than the
  normal approximation because it stays inside [0, 1] and keeps close to nominal
  coverage at the extremes, which is exactly where the interesting counts here
  sit (Brown, Cai & DasGupta, *Statistical Science* 2001).
* **Hanley--McNeil standard errors** for ROC AUC, which need only the AUC and
  the two class sizes (Hanley & McNeil, *Radiology* 1982).

A caveat that belongs with the AUC intervals and is repeated in the docs: the
models here are evaluated on the *same* test set, so their errors are
correlated. Comparing two models by asking whether their Hanley--McNeil
intervals overlap is conservative -- it will miss differences a paired test
would find. A paired DeLong test is the right instrument, and it needs
per-sequence scores, which findings.json does not store yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Two-sided normal quantile for a 95% interval.
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float

    def as_dict(self) -> dict[str, float]:
        return {
            "estimate": round(self.estimate, 4),
            "ci_lower": round(self.lower, 4),
            "ci_upper": round(self.upper, 4),
        }

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.lower:.3f}, {self.upper:.3f}]"


def wilson(successes: int, trials: int, z: float = Z_95) -> Interval:
    """Wilson score interval for a binomial proportion.

    Unlike the normal approximation this never leaves [0, 1] and does not
    collapse to a zero-width interval when ``successes`` is 0 or ``trials`` --
    the cases this project actually reports, such as 0 false positives on 172
    GPCR decoys.

    Raises:
        ValueError: if ``trials`` is not positive or ``successes`` is out of range.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes {successes} outside 0..{trials}")

    proportion = successes / trials
    denominator = 1 + z**2 / trials
    centre = (proportion + z**2 / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(proportion * (1 - proportion) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return Interval(proportion, max(0.0, centre - spread), min(1.0, centre + spread))


def hanley_mcneil(auc: float, n_positive: int, n_negative: int, z: float = Z_95) -> Interval:
    """Confidence interval for a ROC AUC by the Hanley--McNeil method.

    The standard error uses the exponential approximations
    ``Q1 = AUC / (2 - AUC)`` and ``Q2 = 2 AUC^2 / (1 + AUC)`` for the two
    conditional probabilities, which is the usual closed form when only the AUC
    and the class sizes are available.

    Raises:
        ValueError: if the AUC is outside [0, 1] or either class is empty.
    """
    if not 0.0 <= auc <= 1.0:
        raise ValueError(f"auc {auc} outside [0, 1]")
    if n_positive <= 0 or n_negative <= 0:
        raise ValueError("both classes must be non-empty")

    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    variance = (
        auc * (1 - auc) + (n_positive - 1) * (q1 - auc**2) + (n_negative - 1) * (q2 - auc**2)
    ) / (n_positive * n_negative)
    spread = z * math.sqrt(max(variance, 0.0))
    return Interval(auc, max(0.0, auc - spread), min(1.0, auc + spread))


def separated(first: Interval, second: Interval) -> bool:
    """True if two intervals do not overlap.

    Non-overlap implies a difference; overlap does **not** imply the absence of
    one, especially for models scored on the same sequences.
    """
    return first.upper < second.lower or second.upper < first.lower


# The models this project compares, in the order the write-ups present them.
MODEL_LABELS: dict[str, str] = {
    "full_length": "Profile HMM, full-length",
    "tm_region": "Profile HMM, TM regions",
    "esm_logreg": "ESM-2 + logistic regression",
    "esm_mlp": "ESM-2 + MLP",
}

DECOY_CLASSES: tuple[str, ...] = ("type_ii", "gpcr", "globular")


def summarise(findings: dict[str, Any]) -> dict[str, Any]:
    """Attach confidence intervals to every headline number in ``findings``.

    Reads a completed ``findings.json`` and returns the interval report, so the
    statistics can be regenerated without re-running any model. Three things are
    reported, each answering a question the point estimates alone cannot:

    * **AUC intervals**, and which pairs of models are separated by them.
    * **Per-decoy-class false-positive rates**, where the counts are small and a
      rate of 0 needs an upper bound rather than a bare zero.
    * **The within-model decoy ordering** -- whether each model really does
      confuse type II proteins more than GPCRs and globular decoys, which is the
      claim that the errors are biologically structured rather than arbitrary.
    """
    dataset = findings["dataset"]
    assert isinstance(dataset, dict)
    n_positive = int(dataset["positive_test"])
    n_negative = sum(int(v) for v in dict(dataset["negatives_by_class"]).values())

    present = [name for name in MODEL_LABELS if name in findings]

    auc: dict[str, Interval] = {}
    for name in present:
        block = findings[name]
        assert isinstance(block, dict)
        auc[name] = hanley_mcneil(float(block["roc_auc"]), n_positive, n_negative)

    auc_pairs = []
    for i, first in enumerate(present):
        for second in present[i + 1 :]:
            auc_pairs.append(
                {
                    "a": MODEL_LABELS[first],
                    "b": MODEL_LABELS[second],
                    "separated": separated(auc[first], auc[second]),
                }
            )

    rates: dict[str, dict[str, Interval]] = {}
    ordering = []
    for name in present:
        block = findings[name]
        assert isinstance(block, dict)
        by_class = dict(block["false_positives_by_class"])
        per_model = {
            cls: wilson(
                int(dict(by_class[cls])["false_positives"]),
                int(dict(by_class[cls])["sequences"]),
            )
            for cls in DECOY_CLASSES
            if cls in by_class
        }
        rates[name] = per_model
        if "type_ii" in per_model:
            hard = per_model["type_ii"]
            ordering.append(
                {
                    "model": MODEL_LABELS[name],
                    "type_ii_rate": hard.as_dict(),
                    "harder_than": sorted(
                        cls
                        for cls, other in per_model.items()
                        if cls != "type_ii"
                        and hard.estimate > other.estimate
                        and separated(hard, other)
                    ),
                    "not_separated_from": sorted(
                        cls
                        for cls, other in per_model.items()
                        if cls != "type_ii" and not separated(hard, other)
                    ),
                    "confused_more_than_type_ii": sorted(
                        cls
                        for cls, other in per_model.items()
                        if cls != "type_ii"
                        and other.estimate > hard.estimate
                        and separated(other, hard)
                    ),
                }
            )

    return {
        "method": {
            "proportions": "Wilson score interval, 95%",
            "auc": "Hanley-McNeil, 95%",
            "caveat": (
                "Models are scored on the same test set, so their errors are "
                "correlated. Interval overlap is a conservative test for a "
                "difference; a paired DeLong test on the per-sequence scores is "
                "the sharper instrument."
            ),
            "test_positives": n_positive,
            "test_negatives": n_negative,
        },
        "roc_auc": {MODEL_LABELS[k]: v.as_dict() for k, v in auc.items()},
        "roc_auc_pairs": auc_pairs,
        "false_positive_rate": {
            MODEL_LABELS[k]: {cls: iv.as_dict() for cls, iv in v.items()}
            for k, v in rates.items()
        },
        "decoy_difficulty_ordering": ordering,
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Render :func:`summarise` output as Markdown."""
    method = summary["method"]
    lines = [
        f"Test set: {method['test_positives']} positives, "
        f"{method['test_negatives']} decoys. "
        f"AUC intervals {method['auc']}; proportions {method['proportions']}.",
        "",
        "| Model | ROC AUC | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for label, entry in summary["roc_auc"].items():
        lines.append(
            f"| {label} | {entry['estimate']:.4f} | "
            f"[{entry['ci_lower']:.3f}, {entry['ci_upper']:.3f}] |"
        )

    overlapping = [
        f"{pair['a']} vs {pair['b']}"
        for pair in summary["roc_auc_pairs"]
        if not pair["separated"]
    ]
    lines += [
        "",
        "Pairs whose AUC intervals **overlap** (no difference established): "
        + (", ".join(overlapping) if overlapping else "none"),
        "",
        "| Model | type II | GPCR | globular |",
        "| --- | --- | --- | --- |",
    ]
    for label, per_class in summary["false_positive_rate"].items():
        cells = []
        for cls in DECOY_CLASSES:
            values = per_class.get(cls)
            if values is None:
                cells.append("-")
                continue
            cells.append(
                f"{values['estimate']:.3f} "
                f"[{values['ci_lower']:.3f}, {values['ci_upper']:.3f}]"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "Decoy difficulty, within each model:"]
    for row in summary["decoy_difficulty_ordering"]:
        parts = []
        if row["harder_than"]:
            parts.append("type II harder than " + ", ".join(row["harder_than"]))
        if row["confused_more_than_type_ii"]:
            parts.append(
                "**ordering inverted** — "
                + ", ".join(row["confused_more_than_type_ii"])
                + " confused more than type II"
            )
        if row["not_separated_from"]:
            parts.append("not separated from " + ", ".join(row["not_separated_from"]))
        lines.append(f"- **{row['model']}**: " + "; ".join(parts))

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the interval report from a completed ``findings.json``.

    Separate from the pipeline entry point so the statistics can be recomputed
    without the HMM search, the embedding step, or a network call:
    ``python -m tmclass.intervals results/findings.json``.
    """
    import argparse
    import json
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(
        prog="python -m tmclass.intervals",
        description="Confidence intervals for a completed tmclass run.",
    )
    parser.add_argument(
        "findings", type=_Path, nargs="?", default=_Path("results/findings.json")
    )
    parser.add_argument("--output", type=_Path, default=None, help="Write JSON here too.")
    args = parser.parse_args(argv)

    if not args.findings.exists():
        parser.error(f"{args.findings} not found; run the pipeline first")

    report = summarise(json.loads(args.findings.read_text(encoding="utf-8")))
    print(format_summary(report))
    destination = args.output or args.findings.with_name("intervals.json")
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWritten to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
