"""
Final evaluation & baselines.

Given the best hyper-parameters found by the Osprey search this module:

1. retrains the GNN to convergence and evaluates it on the held-out test set,
2. trains several **baselines** for a fair comparison, namely
   * a Random-Forest on the raw node features (classic tabular IDS),
   * a plain MLP on the raw node features (no graph),
   * an *untuned* GNN with default hyper-parameters (isolates the value that the
     Osprey search adds), and
3. writes all metrics, plots and the trained model to ``results/``.

Comparing (3) against the tuned GNN is the experiment that demonstrates the
paper's core claim: Osprey-tuned GNN > untuned GNN > non-graph baselines.
"""
from __future__ import annotations

import numpy as np

from config import CLASS_NAMES, NUM_CLASSES, TrainConfig
from src.graph.builder import GraphData
from src.train import train_gnn
from src.utils import (
    classification_metrics,
    plot_confusion_matrix,
    plot_model_comparison,
)

# A *reasonable, commonly-used* default (NOT a strawman) so the comparison
# against the Osprey-tuned model is fair: 2-layer GCN, moderate lr/dropout.
DEFAULT_GNN_HP = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.5,
    "lr": 5e-3,
    "weight_decay": 5e-4,
    "gnn_type": "gcn",
}


def _random_forest_baseline(graph: GraphData, seed: int) -> dict:
    from sklearn.ensemble import RandomForestClassifier

    tr, te = graph.train_mask | graph.val_mask, graph.test_mask
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(graph.x[tr], graph.y[tr])
    pred = clf.predict(graph.x[te])
    return classification_metrics(graph.y[te], pred, graph.num_classes)


def _mlp_baseline(graph: GraphData, seed: int) -> dict:
    """A graph-blind MLP: same features, no message passing (empty edge set)."""
    import torch

    # reuse the GNN trainer but strip edges -> pure per-node MLP over features
    edgeless = GraphData(
        x=graph.x, y=graph.y,
        edge_index=np.stack([np.arange(graph.num_nodes)] * 2).astype(np.int64),  # self-loops only
        train_mask=graph.train_mask, val_mask=graph.val_mask, test_mask=graph.test_mask,
        num_nodes=graph.num_nodes, num_features=graph.num_features,
        num_classes=graph.num_classes,
    )
    hp = dict(DEFAULT_GNN_HP)
    hp["gnn_type"] = "sage"  # sage with self-loops only == MLP on concatenated self features
    outcome = train_gnn(edgeless, hp, epochs=200, patience=30, seed=seed)
    return outcome.test_metrics


def run_final_evaluation(graph: GraphData, best_hp: dict, tr_cfg: TrainConfig,
                         osprey_result=None, results_dir: str = "results",
                         seed: int = 42, save_model: bool = True) -> dict:
    """Train the tuned GNN, run baselines, write artefacts, return a summary."""
    import os

    import torch

    os.makedirs(results_dir, exist_ok=True)

    # --- 1. Osprey-tuned GNN (final long training) ---
    print("[eval] training Osprey-tuned GNN to convergence ...")
    tuned = train_gnn(graph, best_hp, epochs=tr_cfg.epochs, patience=tr_cfg.patience,
                      seed=seed, verbose=True)

    # --- 2. baselines ---
    print("[eval] training baselines (untuned GNN, RandomForest, MLP) ...")
    untuned = train_gnn(graph, DEFAULT_GNN_HP, epochs=tr_cfg.epochs,
                        patience=tr_cfg.patience, seed=seed)
    rf_metrics = _random_forest_baseline(graph, seed)
    mlp_metrics = _mlp_baseline(graph, seed)

    results = {
        "RandomForest": rf_metrics,
        "MLP (no graph)": mlp_metrics,
        "GNN (untuned)": untuned.test_metrics,
        "GNN (Osprey-tuned)": tuned.test_metrics,
    }

    # --- 3. plots ---
    plot_confusion_matrix(
        tuned.y_test_true, tuned.y_test_pred, CLASS_NAMES,
        os.path.join(results_dir, "confusion_matrix.png"),
        title="Osprey-tuned GNN — Test Confusion Matrix",
    )
    plot_model_comparison(
        results, os.path.join(results_dir, "model_comparison.png"),
        metric="macro_f1", title="Test macro-F1 by model",
    )
    if osprey_result is not None:
        from src.utils import plot_convergence

        plot_convergence(
            osprey_result.history,
            os.path.join(results_dir, "osprey_convergence.png"),
        )

    # --- 4. persist model + summary ---
    if save_model:
        torch.save(
            {"state_dict": tuned.model.state_dict(), "hyperparams": best_hp,
             "feature_dim": graph.num_features, "num_classes": graph.num_classes},
            os.path.join(results_dir, "best_model.pt"),
        )

    summary = {
        "best_hyperparameters": best_hp,
        "results": results,
        "tuned_val_macro_f1": tuned.best_val_f1,
        "osprey_evaluations": getattr(osprey_result, "evaluations", None),
    }
    return summary
