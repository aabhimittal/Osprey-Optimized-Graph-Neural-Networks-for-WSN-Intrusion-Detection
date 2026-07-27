"""
Cross-round trust / reputation scoring.

A single LEACH round gives one noisy verdict per node.  Real compromised nodes,
however, *persist*: a blackhole stays a blackhole round after round.  This
module aggregates the GNN's per-round posteriors into a per-node **trust
index** with an exponential moving average (EMA):

    trust_t(v) = alpha * trust_{t-1}(v) + (1 - alpha) * P(normal | v, round t)

starting from trust_0 = 1 (innocent until proven otherwise).  A node whose
trust falls below ``threshold`` is quarantined (e.g. excluded from CH election
— the natural WSN response).

Why this beats per-round detection for persistent attackers: single-round
false negatives are transient, but the EMA integrates evidence, so a node that
keeps behaving badly is driven monotonically toward zero trust while honest
nodes hover near 1.  The module quantifies exactly that: it compares
**per-round classification** vs **trust-based flagging** for the persistent
attackers planted by the generator.

Outputs: ``results/trust_scores.png`` (trajectories) and
``results/trust_report.json``.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from config import CLASS_NAMES
from src.data.loader import Dataset
from src.graph.builder import GraphData
from src.utils import ensure_dir, save_json


def compute_trust(model, graph: GraphData, ds: Dataset, alpha: float = 0.7,
                  device: str = "cpu") -> dict:
    """Return per-node trust trajectories over rounds.

    Returns dict with:
      ``rounds``      – sorted round ids,
      ``node_ids``    – sorted physical node ids,
      ``trust``       – (n_nodes, n_rounds) trust trajectory matrix,
      ``p_normal``    – raw per-round P(normal) matrix (NaN when node absent),
      ``mal_fraction``– per node, fraction of rounds it was truly malicious.
    """
    model.eval()
    x = torch.tensor(graph.x, dtype=torch.float32, device=device)
    ei = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    with torch.no_grad():
        probs = torch.softmax(model(x, ei), dim=-1).cpu().numpy()
    p_normal_flat = probs[:, 0]                     # class 0 = Normal

    rounds = np.sort(np.unique(ds.round_id))
    node_ids = np.sort(np.unique(ds.node_id))
    r_index = {r: i for i, r in enumerate(rounds)}
    n_index = {n: i for i, n in enumerate(node_ids)}

    p_normal = np.full((len(node_ids), len(rounds)), np.nan)
    was_mal = np.zeros((len(node_ids), len(rounds)), dtype=bool)
    for k in range(len(ds.y)):
        i, j = n_index[ds.node_id[k]], r_index[ds.round_id[k]]
        p_normal[i, j] = p_normal_flat[k]
        was_mal[i, j] = ds.y[k] != 0

    # EMA over rounds; carry trust through rounds where the node is absent
    trust = np.ones((len(node_ids), len(rounds)))
    prev = np.ones(len(node_ids))
    for j in range(len(rounds)):
        obs = p_normal[:, j]
        cur = np.where(np.isnan(obs), prev, alpha * prev + (1 - alpha) * obs)
        trust[:, j] = cur
        prev = cur

    return {
        "rounds": rounds,
        "node_ids": node_ids,
        "trust": trust,
        "p_normal": p_normal,
        "mal_fraction": was_mal.mean(axis=1),
    }


def evaluate_trust(tr: dict, threshold: float = 0.5,
                   persistent_cutoff: float = 0.8) -> dict:
    """Score trust-based flagging of *persistent* attackers.

    A node is ground-truth "compromised" if it was malicious in at least
    ``persistent_cutoff`` of its rounds; it is flagged if its final trust is
    below ``threshold``.  Also reports the mean detection latency: the first
    round at which a compromised node's trust crossed the threshold.
    """
    final_trust = tr["trust"][:, -1]
    compromised = tr["mal_fraction"] >= persistent_cutoff
    flagged = final_trust < threshold

    tp = int(np.sum(flagged & compromised))
    fp = int(np.sum(flagged & ~compromised))
    fn = int(np.sum(~flagged & compromised))
    tn = int(np.sum(~flagged & ~compromised))

    # detection latency (in rounds) for the caught compromised nodes
    latencies = []
    for i in np.where(compromised & flagged)[0]:
        below = np.where(tr["trust"][i] < threshold)[0]
        if below.size:
            latencies.append(int(below[0]) + 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "n_compromised": int(compromised.sum()),
        "n_flagged": int(flagged.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "mean_detection_latency_rounds": float(np.mean(latencies)) if latencies else None,
        "threshold": threshold,
        "persistent_cutoff": persistent_cutoff,
    }


def plot_trust(tr: dict, path: str, threshold: float = 0.5, n_show: int = 12):
    """Plot trust trajectories: compromised nodes in red, honest in green."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    compromised = tr["mal_fraction"] >= 0.8
    honest = tr["mal_fraction"] <= 0.05
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for i in rng.choice(np.where(honest)[0], size=min(n_show, honest.sum()), replace=False):
        ax.plot(tr["rounds"], tr["trust"][i], color="#2d6a4f", alpha=0.5, lw=1.2)
    for i in rng.choice(np.where(compromised)[0], size=min(n_show, compromised.sum()), replace=False):
        ax.plot(tr["rounds"], tr["trust"][i], color="#c1121f", alpha=0.75, lw=1.6)
    ax.axhline(threshold, color="#555", ls="--", lw=1, label=f"quarantine threshold ({threshold})")
    ax.set_xlabel("LEACH round")
    ax.set_ylabel("Trust index")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Node trust over time — honest (green) vs persistent attackers (red)")
    ax.legend(loc="center right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_trust(model, graph: GraphData, ds: Dataset, results_dir: str = "results",
              alpha: float = 0.7, threshold: float = 0.5) -> dict:
    tr = compute_trust(model, graph, ds, alpha=alpha)
    report = evaluate_trust(tr, threshold=threshold)

    # reference: how well does a *single* round do on the same nodes?  Use the
    # last round's raw posterior as a one-shot detector.
    one_shot_flag = tr["p_normal"][:, -1] < threshold
    compromised = tr["mal_fraction"] >= 0.8
    valid = ~np.isnan(tr["p_normal"][:, -1])
    os_tp = int(np.sum(one_shot_flag[valid] & compromised[valid]))
    os_fp = int(np.sum(one_shot_flag[valid] & ~compromised[valid]))
    os_fn = int(np.sum(~one_shot_flag[valid] & compromised[valid]))
    os_p = os_tp / max(os_tp + os_fp, 1)
    os_r = os_tp / max(os_tp + os_fn, 1)
    report["single_round_reference"] = {
        "precision": os_p, "recall": os_r,
        "f1": 2 * os_p * os_r / max(os_p + os_r, 1e-12),
    }

    plot_trust(tr, os.path.join(results_dir, "trust_scores.png"), threshold=threshold)
    save_json(report, os.path.join(results_dir, "trust_report.json"))
    return report
