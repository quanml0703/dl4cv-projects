"""Evaluation utilities for humanitarian crisis classification experiments.

Provides:
- Normalised confusion-matrix visualisation (seaborn).
- Macro-F1, weighted-F1, and macro-OvR AUC computation.
- Per-class classification report (precision / recall / F1).
- Side-by-side experiment comparison table and bar chart.
- Inference-time comparison utilities.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

#: Canonical order of the 5 final labels (used for all matrix / report outputs).
LABELS: list[str] = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
]

#: Short display names for axis tick labels in plots.
SHORT_LABELS: list[str] = [
    "Affected\nIndividuals",
    "Infra.\nDamage",
    "Not\nHumanitarian",
    "Other\nRelevant",
    "Rescue &\nVolunteering",
]


# ─────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    title: str = "Confusion Matrix",
    ax: Optional[plt.Axes] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot a confusion matrix using a seaborn heatmap.

    Args:
        y_true:    Ground-truth label list.
        y_pred:    Predicted label list.
        title:     Plot title.
        ax:        Existing ``matplotlib.Axes`` to draw on (creates a new
                   figure when ``None``).
        normalize: When ``True``, display row-normalised (recall) values.

    Returns:
        The ``matplotlib.Figure`` containing the heatmap.
    """
    norm = "true" if normalize else None
    cm   = confusion_matrix(y_true, y_pred, labels=LABELS, normalize=norm)
    fmt  = ".2f" if normalize else "d"

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=SHORT_LABELS,
        yticklabels=SHORT_LABELS,
        linewidths=0.4,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> dict[str, float]:
    """Compute macro-F1, weighted-F1, and macro one-vs-rest AUC.

    Note: AUC is computed from binarised hard predictions (not probabilities),
    so it serves as an indicator of discriminative ability rather than a
    calibrated probability estimate.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict with keys ``'f1_macro'``, ``'f1_weighted'``, ``'auc_macro'``.
    """
    f1_macro    = f1_score(y_true, y_pred, average="macro",    labels=LABELS, zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=LABELS, zero_division=0)

    # Binarise for multi-class AUC (OvR)
    y_true_bin = label_binarize(y_true, classes=LABELS)
    y_pred_bin = label_binarize(y_pred, classes=LABELS)
    try:
        auc = roc_auc_score(
            y_true_bin, y_pred_bin, average="macro", multi_class="ovr"
        )
    except ValueError:
        auc = float("nan")

    return {
        "f1_macro":    round(f1_macro,    4),
        "f1_weighted": round(f1_weighted, 4),
        "auc_macro":   round(auc,         4),
    }


def print_classification_report(
    y_true: list[str],
    y_pred: list[str],
) -> None:
    """Print a per-class precision / recall / F1 / support report.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
    """
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))


# ─────────────────────────────────────────────────────────
# Comparison utilities
# ─────────────────────────────────────────────────────────

def build_comparison_df(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Assemble per-experiment metric dicts into a tidy comparison DataFrame.

    Args:
        results: Mapping of ``experiment_name`` → metrics dict
                 (output of :func:`compute_metrics`).

    Returns:
        DataFrame indexed by experiment name.
    """
    rows = [{"experiment": name, **metrics} for name, metrics in results.items()]
    return pd.DataFrame(rows).set_index("experiment")


def plot_metric_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    title: str = "Experiment Comparison",
) -> plt.Figure:
    """Plot a grouped bar chart comparing experiments across metrics.

    Args:
        comparison_df: Output of :func:`build_comparison_df`.
        metrics:       Column names to plot; defaults to all numeric columns.
        title:         Plot title.

    Returns:
        The ``matplotlib.Figure``.
    """
    if metrics is None:
        metrics = comparison_df.select_dtypes("number").columns.tolist()

    plot_df = (
        comparison_df[metrics]
        .reset_index()
        .melt(id_vars="experiment", var_name="metric", value_name="score")
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="experiment", y="score", hue="metric", ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=15)

    # Annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    return fig


def plot_inference_time(
    results: dict[str, list[float]],
    title: str = "Inference Time per Sample",
) -> plt.Figure:
    """Plot a box plot of per-sample inference times across experiments.

    Args:
        results: Mapping of ``experiment_name`` → list of inference times (seconds).
        title:   Plot title.

    Returns:
        The ``matplotlib.Figure``.
    """
    records = [
        {"experiment": name, "inference_time_s": t}
        for name, times in results.items()
        for t in times
    ]
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="experiment", y="inference_time_s", ax=ax, palette="Set2")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Inference time (s)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    return fig
