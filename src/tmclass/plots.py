"""ROC, precision-recall and confusion-matrix figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_COLOURS = {
    "full_length": "#c8ced4",
    "tm_region": "#1f77b4",
    "esm_logreg": "#e8a33d",
    "esm_mlp": "#4f9d69",
}
MODEL_LABELS = {
    "full_length": "Profile HMM, full-length",
    "tm_region": "Profile HMM, TM regions",
    "esm_logreg": "ESM-2 + logistic regression",
    "esm_mlp": "ESM-2 + MLP",
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
    columns = min(len(models), 2)
    rows = (len(models) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.4 * rows))
    axes = [axes] if len(models) == 1 else list(axes.flatten())

    for extra in axes[len(models) :]:
        extra.set_visible(False)

    # axes is padded to a full grid, so it can be longer than models.
    for ax, model in zip(axes, models, strict=False):
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
                    j,
                    i,
                    f"{matrix[i][j]}",
                    ha="center",
                    va="center",
                    fontsize=15,
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


def plot_model_comparison(findings: dict, path: Path) -> Path:
    """Headline bar chart: every model on the metrics that survive imbalance."""
    models = _models(findings)
    metrics = [
        ("roc_auc", "ROC AUC", lambda f: f["roc_auc"]),
        ("average_precision", "Avg. precision", lambda f: f["average_precision"]),
        ("mcc", "MCC", lambda f: f["at_optimal_cutoff"]["mcc"]),
        ("sensitivity", "Sensitivity", lambda f: f["at_optimal_cutoff"]["sensitivity"]),
        ("precision", "Precision", lambda f: f["at_optimal_cutoff"]["precision"]),
    ]

    positions = range(len(metrics))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for index, model in enumerate(models):
        offset = (index - (len(models) - 1) / 2) * width
        values = [getter(findings[model]) for _, _, getter in metrics]
        bars = ax.bar(
            [p + offset for p in positions],
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLOURS[model],
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(list(positions))
    ax.set_xticklabels([label for _, label, _ in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(
        "Profile HMM versus ESM-2 embeddings on the same held-out test set",
        fontsize=11,
    )
    ax.legend(
        frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.10)
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eef1f3")
    ax.set_axisbelow(True)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_by_decoy_class(findings: dict, path: Path) -> Path:
    """False-positive rate per decoy class, which is where the biology shows."""
    models = _models(findings)
    classes = list(findings[models[0]]["false_positives_by_class"])

    positions = range(len(classes))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for index, model in enumerate(models):
        offset = (index - (len(models) - 1) / 2) * width
        by_class = findings[model]["false_positives_by_class"]
        values = [100 * by_class[name]["false_positive_rate"] for name in classes]
        bars = ax.bar(
            [p + offset for p in positions],
            values,
            width,
            label=MODEL_LABELS[model],
            color=MODEL_COLOURS[model],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    labels = []
    for name in classes:
        entry = findings[models[-1]]["false_positives_by_class"][name]
        suffix = "\n(never trained on)" if entry.get("held_out_from_training") else ""
        labels.append(name.replace("_", " ") + suffix)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("False-positive rate (%)")
    ax.set_title(
        "Where each model makes its mistakes, at its own optimal threshold",
        fontsize=11,
    )
    ax.legend(
        frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12)
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eef1f3")
    ax.set_axisbelow(True)
    fig.tight_layout()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
