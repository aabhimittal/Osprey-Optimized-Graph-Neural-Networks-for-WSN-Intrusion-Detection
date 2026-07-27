"""
Explainability for the trained IDS.

An intrusion-detection alarm is only actionable if an operator can see *why* a
node was flagged.  This module provides two complementary, model-agnostic
explanations for the trained GNN:

1. **Feature saliency (what).**  For every test node we compute the gradient of
   its predicted-class logit w.r.t. the *input feature matrix* and take the
   absolute gradient row belonging to that node.  Averaging per predicted class
   yields a ``classes x features`` saliency map: e.g. Blackhole predictions
   should light up ``Forward_Ratio`` / ``DATA_R``, Flooding should light up
   ``ADV_S``.  This doubles as a sanity check that the model learned the actual
   attack signatures rather than shortcuts.

2. **Neighbour occlusion (who).**  For a sample of flagged nodes we remove one
   neighbour's edges at a time and measure the drop in the predicted-class
   logit.  A large drop means that neighbour's messages were important evidence
   — in a WSN this identifies e.g. *which cluster members' traffic exposed a
   blackhole CH*.

Outputs: ``results/explainability.png`` (saliency heatmap) and
``results/explanations.json`` (top features per class + per-node neighbour
attributions).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from config import CLASS_NAMES, FEATURE_NAMES
from src.graph.builder import GraphData
from src.utils import ensure_dir, save_json


# --------------------------------------------------------------------------- #
# 1. Gradient saliency
# --------------------------------------------------------------------------- #
def feature_saliency(model, graph: GraphData, mask: np.ndarray | None = None,
                     device: str = "cpu") -> np.ndarray:
    """Mean |d logit_pred / d x| per (predicted class, feature).

    Returns an array of shape ``(num_classes, num_features)`` normalised so each
    class row sums to 1 (a per-class feature-importance distribution).
    """
    model.eval()
    x = torch.tensor(graph.x, dtype=torch.float32, device=device, requires_grad=True)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)

    logits = model(x, edge_index)
    preds = logits.argmax(dim=-1)

    sel = np.where(mask if mask is not None else np.ones(graph.num_nodes, bool))[0]

    sal = np.zeros((graph.num_classes, graph.num_features))
    counts = np.zeros(graph.num_classes)

    # one backward pass per class (not per node): sum the predicted-class logits
    # of all selected nodes predicted as c, then read each node's own grad row.
    for c in range(graph.num_classes):
        nodes_c = sel[preds[sel].cpu().numpy() == c]
        if len(nodes_c) == 0:
            continue
        if x.grad is not None:
            x.grad = None
        logits[nodes_c, c].sum().backward(retain_graph=True)
        g = x.grad.detach().cpu().numpy()
        sal[c] = np.abs(g[nodes_c]).mean(axis=0)
        counts[c] = len(nodes_c)

    # normalise rows to distributions
    row_sums = sal.sum(axis=1, keepdims=True)
    sal = np.divide(sal, row_sums, out=np.zeros_like(sal), where=row_sums > 0)
    return sal


# --------------------------------------------------------------------------- #
# 2. Neighbour occlusion
# --------------------------------------------------------------------------- #
@torch.no_grad()
def neighbour_occlusion(model, graph: GraphData, node: int,
                        device: str = "cpu") -> list[dict]:
    """Rank a node's neighbours by how much removing them lowers its logit.

    Returns a list of ``{"neighbour": id, "logit_drop": float}`` sorted by
    importance (largest drop first).
    """
    model.eval()
    x = torch.tensor(graph.x, dtype=torch.float32, device=device)
    ei = torch.tensor(graph.edge_index, dtype=torch.long, device=device)

    base_logits = model(x, ei)
    pred_class = int(base_logits[node].argmax())
    base = float(base_logits[node, pred_class])

    src, dst = graph.edge_index
    neighbours = np.unique(np.concatenate([dst[src == node], src[dst == node]]))
    neighbours = neighbours[neighbours != node]

    out = []
    for nb in neighbours:
        keep = ~(((src == nb) & (dst == node)) | ((src == node) & (dst == nb)))
        ei_occ = torch.tensor(graph.edge_index[:, keep], dtype=torch.long, device=device)
        occluded = float(model(x, ei_occ)[node, pred_class])
        out.append({"neighbour": int(nb), "logit_drop": base - occluded})

    out.sort(key=lambda d: -d["logit_drop"])
    return out


# --------------------------------------------------------------------------- #
# Orchestration + plotting
# --------------------------------------------------------------------------- #
def plot_saliency(sal: np.ndarray, path: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.heatmap(sal, cmap="magma", xticklabels=FEATURE_NAMES,
                yticklabels=CLASS_NAMES,
                cbar_kws={"label": "feature importance (per-class share)"}, ax=ax)
    ax.set_title("What the IDS looks at — gradient saliency per predicted class")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_explain(model, graph: GraphData, results_dir: str = "results",
                n_occlusion_nodes: int = 10, seed: int = 42) -> dict:
    """Full explainability pass: saliency map + occlusion for sample attack nodes."""
    rng = np.random.default_rng(seed)

    sal = feature_saliency(model, graph, mask=graph.test_mask)
    plot_saliency(sal, os.path.join(results_dir, "explainability.png"))

    # top-3 features per class
    top_features = {
        CLASS_NAMES[c]: [FEATURE_NAMES[i] for i in np.argsort(-sal[c])[:3]]
        for c in range(len(CLASS_NAMES))
    }

    # occlusion on a sample of test nodes predicted malicious
    x = torch.tensor(graph.x, dtype=torch.float32)
    ei = torch.tensor(graph.edge_index, dtype=torch.long)
    preds = model.predict(x, ei).numpy()
    flagged = np.where(graph.test_mask & (preds > 0))[0]
    sample = rng.choice(flagged, size=min(n_occlusion_nodes, len(flagged)), replace=False)

    occl = {}
    for node in sample:
        occl[int(node)] = {
            "predicted": CLASS_NAMES[int(preds[node])],
            "true": CLASS_NAMES[int(graph.y[node])],
            "top_neighbours": neighbour_occlusion(model, graph, int(node))[:5],
        }

    report = {"top_features_per_class": top_features, "neighbour_occlusion": occl}
    save_json(report, os.path.join(results_dir, "explanations.json"))
    return report
