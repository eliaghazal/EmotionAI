"""Plotting utilities: training curves, confusion matrices, per-class bars."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

EMOTION_LABELS: List[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]
EMOTION_COLORS: List[str] = [
    "#FF4444", "#9B59B6", "#E67E22", "#F1C40F",
    "#3498DB", "#1ABC9C", "#95A5A6",
]

FIGURE_DPI = 150


def plot_training_curves(
    histories: Dict[str, object],
    save_path: str,
) -> None:
    """Plot accuracy and loss curves for one or more training histories.

    Args:
        histories: Dict mapping model name → Keras History object.
        save_path: File path for the saved PNG.
    """
    n_models = len(histories)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 5 * n_models))
    if n_models == 1:
        axes = [axes]

    plt.style.use("seaborn-v0_8-darkgrid")
    for row, (name, history) in enumerate(histories.items()):
        h = history.history
        epochs = range(1, len(h["accuracy"]) + 1)

        axes[row][0].plot(epochs, h["accuracy"], label="Train Acc", color="#3498DB", lw=2)
        axes[row][0].plot(epochs, h["val_accuracy"], label="Val Acc", color="#1ABC9C", lw=2, ls="--")
        axes[row][0].set_title(f"{name} — Accuracy")
        axes[row][0].set_xlabel("Epoch")
        axes[row][0].set_ylabel("Accuracy")
        axes[row][0].legend()

        axes[row][1].plot(epochs, h["loss"], label="Train Loss", color="#FF4444", lw=2)
        axes[row][1].plot(epochs, h["val_loss"], label="Val Loss", color="#E67E22", lw=2, ls="--")
        axes[row][1].set_title(f"{name} — Loss")
        axes[row][1].set_xlabel("Epoch")
        axes[row][1].set_ylabel("Loss")
        axes[row][1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved to %s", save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str,
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Integer ground-truth labels, shape (N,).
        y_pred: Integer predicted labels, shape (N,).
        model_name: Title prefix for the plot.
        save_path: File path for the saved PNG.
        normalize: Whether to show row-normalized percentages.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = f"{model_name} — Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = f"{model_name} — Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", save_path)


def plot_per_class_metrics(
    report_dict: Dict[str, Dict[str, float]],
    model_name: str,
    save_path: str,
) -> None:
    """Bar chart of per-class precision, recall, and F1-score.

    Args:
        report_dict: Output of sklearn classification_report(output_dict=True).
        model_name: Title prefix.
        save_path: File path for the saved PNG.
    """
    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(EMOTION_LABELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        vals = [report_dict.get(label, {}).get(metric, 0) for label in EMOTION_LABELS]
        ax.bar(x + i * width, vals, width, label=metric.capitalize())

    ax.set_title(f"{model_name} — Per-Class Metrics", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(EMOTION_LABELS, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    logger.info("Per-class metrics chart saved to %s", save_path)


def plot_emotion_distribution(
    counts: Dict[str, float],
    title: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Donut chart of emotion distribution for a session.

    Args:
        counts: Dict mapping emotion label → count or percentage.
        title: Chart title.
        save_path: Optional file path to save PNG; if None, figure is returned.

    Returns:
        Matplotlib Figure object.
    """
    labels = list(counts.keys())
    sizes = list(counts.values())
    colors = [EMOTION_COLORS[EMOTION_LABELS.index(l)] if l in EMOTION_LABELS else "#AAAAAA"
              for l in labels]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title(title, fontsize=13, pad=14)

    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()
        logger.info("Distribution chart saved to %s", save_path)
    return fig
