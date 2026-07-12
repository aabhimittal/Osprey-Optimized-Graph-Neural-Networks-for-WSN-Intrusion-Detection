"""Shared utilities: reproducible seeding, metric computation and plotting."""
from __future__ import annotations

import json
import os
import random
from typing import Dict

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (if available) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic-ish CPU behaviour
        torch.use_deterministic_algorithms(False)
    except Exception:  # pragma: no cover - torch always present in practice
        pass


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(y_true, y_pred, num_classes: int) -> Dict[str, float]:
    """Return a dict of the standard IDS metrics.

    Uses macro averaging (every attack class counts equally regardless of how
    rare it is) which is the fair way to score an imbalanced intrusion-detection
    problem.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }


def save_json(obj, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# Plotting (all figures are written to disk; nothing is shown interactively)
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, y_pred, class_names, path: str, title: str = "Confusion Matrix"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Row-normalised rate"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(history, path: str, title: str = "Osprey Optimization Convergence"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    iters = np.arange(1, len(history) + 1)
    ax.plot(iters, history, marker="o", color="#2a6f97", lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best validation macro-F1")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(results: Dict[str, Dict[str, float]], path: str,
                          metric: str = "macro_f1",
                          title: str = "Model comparison (macro-F1)"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    scores = [results[n][metric] for n in names]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, scores, color=colors)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.01, f"{s:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)
