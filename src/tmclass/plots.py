"""ROC, precision-recall and confusion-matrix figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_COLOURS = {"full_length": "#b8c4cc", "tm_region": "#1f77b4"}
MODEL_LABELS = {
    "full_length": "Full-length sequences",
    "tm_region": "TM regions only",
}
RED = "#b4413c"


def _models(findings: dict) -> list[str]:
    return [key for key in MODEL_LABELS if key in findings]


def plot_curves(findings: dict, path: Path) -> Path:
    """ROC and precision-recall curves for every model variant."""
    models = _models(findings)
    fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(11, 4.6))

    for model in models:
        curve = findings[model]["curve"]
        colour = MODEL_COLOURS[model]
        label = MODEL_LABELS[model]

        points = sorted({(p["fpr"], p["tpr"]) for p in curve})
        roc_ax.plot(
            [0, *[x for x, _ in points], 1],
            [0, *[y for _, y in points], 1],
            color=colour,
            linewidth=2,
            label=f"{label} (AUC = {findings[model]['roc_auc']:.3f})",
        )

        pr_points = sorted({(p["tpr"], p["precision"]) for p in curve})
        pr_ax.plot(
            [x for x, _ in pr_points],
            [y for _, y in pr_points],
            color=colour,
            linewidth=2,
            label=f"{label} (AP = {findings[model]['average_precision']:.3f})",
        )

    roc_ax.plot([0, 1], [0, 1], color=RED, linestyle="--", linewidth=1, label="chance")
    roc_ax.set_xlabel("False-positive rate (1 − specificity)")
    roc_ax.set_ylabel("True-positive rate (sensitivity)")
    roc_ax.set_title("ROC", fontsize=11)
    roc_ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    positives = findings["dataset"]["positive_test"]
    negatives = sum(findings["dataset"]["negatives_by_class"].values())
    baseline = positives / (positives + negatives)
    pr_ax.axhline(
        baseline,
        color=RED,
        linestyle="--",
        linewidth=1,
        label=f"chance ({baseline:.2f})",
    )
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_title("Precision-recall", fontsize=11)
    pr_ax.set_ylim(0, 1.05)
    pr_ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    for ax in (roc_ax, pr_ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, 1)
        ax.grid(color="#eef1f3")
        ax.set_axisbelow(True)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_confusion(findings: dict, path: Path) -> Path:
    """Confusion matrices at the MCC-optimal threshold, one per model."""
    models = _models(findings)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        optimal = findings[model]["at_optimal_cutoff"]
        matrix = [
            [optimal["true_positives"], optimal["false_negatives"]],
            [optimal["false_positives"], optimal["true_negatives"]],
        ]
        total = sum(sum(row) for row in matrix)

        ax.imshow(
            [[value / total for value in row] for row in matrix],
            cmap="Blues",
            vmin=0,
            vmax=1,
        )
        for i in range(2):
            for j in range(2):
                share = matrix[i][j] / total
                ax.text(
                    j, i, f"{matrix[i][j]}",
                    ha="center", va="center", fontsize=15,
                    color="white" if share > 0.4 else "#1c2b36",
                )

        ax.set_xticks([0, 1], ["predicted\ntype I", "predicted\nother"])
        ax.set_yticks([0, 1], ["actual\ntype I", "actual\nother"])
        ax.set_title(
            f"{MODEL_LABELS[model]}\nMCC {optimal['mcc']:.3f} · "
            f"precision {optimal['precision']:.3f}",
            fontsize=10,
        )
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
