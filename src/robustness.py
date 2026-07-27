"""
Adversarial / noise robustness evaluation.

A real attacker does not have to match the training distribution: it can jitter
its counters (send a *few* packets to the BS, randomise timing) to slip past a
brittle detector, and radio noise perturbs honest features too.  This module
stress-tests the trained detectors by perturbing the (standard-scaled) test
features and tracking macro-F1 as the perturbation grows:

* **Gaussian noise** — ``x' = x + eps * N(0, 1)`` on every test node
  (models sensor/radio noise).
* **Evasion** — the same noise applied **only to malicious test nodes**,
  which mimics attackers deliberately blurring their own signature while the
  rest of the network stays clean.

The GNN and the Random-Forest baseline are evaluated side by side.  The
structural hypothesis: because the GNN also reads *neighbourhood* evidence, a
malicious node cannot fully hide by editing its own row — its neighbours still
carry the missing-traffic signal — so its curve should degrade more gracefully
under evasion than a purely tabular model's.

Outputs: ``results/robustness.png`` and ``results/robustness.json``.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from src.graph.builder import GraphData
from src.utils import classification_metrics, ensure_dir, save_json

EPSILONS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]


def _perturb(x: np.ndarray, mask: np.ndarray, eps: float,
             rng: np.random.Generator) -> np.ndarray:
    """Add eps-scaled Gaussian noise to the rows of ``x`` selected by ``mask``."""
    x2 = x.copy()
    x2[mask] += eps * rng.standard_normal((int(mask.sum()), x.shape[1])).astype(x.dtype)
    return x2


def _gnn_f1(model, graph: GraphData, x: np.ndarray) -> float:
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32),
                       torch.tensor(graph.edge_index, dtype=torch.long))
    pred = logits[graph.test_mask].argmax(-1).numpy()
    return classification_metrics(graph.y[graph.test_mask], pred, graph.num_classes)["macro_f1"]


def run_robustness(model, graph: GraphData, results_dir: str = "results",
                   seed: int = 42) -> dict:
    """Sweep perturbation strengths for GNN + RandomForest; plot & save."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(seed)
    model.eval()

    # RandomForest trained on clean train+val features (same as evaluate.py)
    tr_mask = graph.train_mask | graph.val_mask
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(graph.x[tr_mask], graph.y[tr_mask])

    test_mask = graph.test_mask
    mal_test_mask = test_mask & (graph.y != 0)

    curves = {"gnn_noise": [], "rf_noise": [], "gnn_evasion": [], "rf_evasion": []}
    for eps in EPSILONS:
        # scenario 1: noise on every test node
        xn = _perturb(graph.x, test_mask, eps, rng)
        curves["gnn_noise"].append(_gnn_f1(model, graph, xn))
        rf_pred = rf.predict(xn[test_mask])
        curves["rf_noise"].append(
            classification_metrics(graph.y[test_mask], rf_pred, graph.num_classes)["macro_f1"])

        # scenario 2: evasion — only malicious nodes perturb themselves
        xe = _perturb(graph.x, mal_test_mask, eps, rng)
        curves["gnn_evasion"].append(_gnn_f1(model, graph, xe))
        rf_pred = rf.predict(xe[test_mask])
        curves["rf_evasion"].append(
            classification_metrics(graph.y[test_mask], rf_pred, graph.num_classes)["macro_f1"])

    report = {"epsilons": EPSILONS, **{k: [float(v) for v in vs] for k, vs in curves.items()}}
    save_json(report, os.path.join(results_dir, "robustness.json"))
    _plot(report, os.path.join(results_dir, "robustness.png"))
    return report


def _plot(report: dict, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = report["epsilons"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)

    axes[0].plot(eps, report["gnn_noise"], "o-", color="#2a6f97", label="Osprey-GNN")
    axes[0].plot(eps, report["rf_noise"], "s--", color="#bc4749", label="RandomForest")
    axes[0].set_title("Gaussian noise on all test nodes")

    axes[1].plot(eps, report["gnn_evasion"], "o-", color="#2a6f97", label="Osprey-GNN")
    axes[1].plot(eps, report["rf_evasion"], "s--", color="#bc4749", label="RandomForest")
    axes[1].set_title("Evasion: only attackers perturb their features")

    for ax in axes:
        ax.set_xlabel("perturbation strength ε (in std units)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("test macro-F1")
    fig.suptitle("Robustness under feature perturbation")
    plt.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)
