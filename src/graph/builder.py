"""
Graph construction for WSN intrusion detection.

Why a graph?  An attack such as a blackhole is only *obvious relative to its
neighbours* — a node that receives lots of data but forwards none looks
suspicious precisely because the honest CHs around it behave differently.  A GNN
exploits exactly this by passing messages along the communication topology, so
we must first turn each LEACH round into a graph.

For every round we build a graph whose

* **nodes**  = the sensor nodes active in that round (features = behavioural
  counters), and
* **edges**  =
    1. *cluster edges*  – every member <-> its Cluster Head (the LEACH tree),
    2. *CH backbone*    – Cluster Heads are inter-connected (they all talk to
       the Base Station, so they share context), and
    3. *spatial k-NN*   – each node is linked to its ``k`` nearest neighbours,
       capturing physical proximity / radio range.

All per-round graphs are then packed into **one disjoint-union graph**
(block-diagonal adjacency): node indices are offset per round so message passing
never leaks across rounds, yet the whole dataset can be trained full-batch with
a single sparse ``edge_index``.  This is the standard, efficient PyG-style
mini-batching, implemented here without the PyG dependency.

The train/val/test split is done **by round** (whole graphs go to one split) so
there is no information leakage between a node and its own neighbours across the
split boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.loader import Dataset


@dataclass
class GraphData:
    x: np.ndarray            # (N, F) node features
    y: np.ndarray            # (N,)   labels
    edge_index: np.ndarray   # (2, E) directed edges (both directions stored)
    train_mask: np.ndarray   # (N,) bool
    val_mask: np.ndarray     # (N,) bool
    test_mask: np.ndarray    # (N,) bool
    num_nodes: int
    num_features: int
    num_classes: int


def _knn_edges(pos: np.ndarray, k: int) -> np.ndarray:
    """Undirected k-NN edges (both directions) among the given positions."""
    n = pos.shape[0]
    if n <= 1:
        return np.empty((2, 0), dtype=np.int64)
    k = min(k, n - 1)
    # pairwise distances
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :k]        # (n, k)
    src = np.repeat(np.arange(n), k)
    dst = nn.reshape(-1)
    # store both directions -> undirected
    e = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    return e.astype(np.int64)


def _round_edges(local_ids, who_ch_local, is_ch, pos, k) -> np.ndarray:
    """Build the edge list for a single round in *local* node indices."""
    n = len(local_ids)
    edges = []

    # 1. cluster (member <-> CH) edges
    id_to_local = {nid: i for i, nid in enumerate(local_ids)}
    for i in range(n):
        ch = who_ch_local[i]
        if ch in id_to_local and id_to_local[ch] != i:
            j = id_to_local[ch]
            edges.append((i, j))
            edges.append((j, i))

    # 2. CH backbone: fully connect the (few) cluster heads
    ch_locals = [i for i in range(n) if is_ch[i] == 1]
    for a in range(len(ch_locals)):
        for b in range(a + 1, len(ch_locals)):
            edges.append((ch_locals[a], ch_locals[b]))
            edges.append((ch_locals[b], ch_locals[a]))

    tree_e = (
        np.array(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    )

    # 3. spatial k-NN edges
    knn_e = _knn_edges(pos, k)

    all_e = np.concatenate([tree_e, knn_e], axis=1) if knn_e.size else tree_e
    return all_e


def build_graph(ds: Dataset, knn_k: int = 6, val_fraction: float = 0.2,
                test_fraction: float = 0.2, seed: int = 42,
                add_self_loops: bool = True) -> GraphData:
    """Assemble the disjoint-union graph and the round-wise split masks."""
    rng = np.random.default_rng(seed)
    rounds = np.unique(ds.round_id)

    # --- assign each round to a split ------------------------------------- #
    shuffled = rng.permutation(rounds)
    n_test = max(1, int(test_fraction * len(rounds)))
    n_val = max(1, int(val_fraction * len(rounds)))
    test_rounds = set(shuffled[:n_test].tolist())
    val_rounds = set(shuffled[n_test:n_test + n_val].tolist())

    N = ds.X.shape[0]
    all_edges = []
    offset = 0
    train_mask = np.zeros(N, dtype=bool)
    val_mask = np.zeros(N, dtype=bool)
    test_mask = np.zeros(N, dtype=bool)

    # deduplicate edges via a set of the global (src,dst) pairs
    for r in rounds:
        sel = np.where(ds.round_id == r)[0]
        local_ids = ds.node_id[sel]
        who_ch_local = ds.who_ch[sel]
        # Is_CH lives inside the feature matrix, but we kept it human-readable in
        # loader too; recompute from who_ch (a node that is its own CH is a CH).
        is_ch = (who_ch_local == local_ids).astype(int)
        pos = ds.pos[sel]

        e_local = _round_edges(local_ids, who_ch_local, is_ch, pos, knn_k)
        if e_local.size:
            all_edges.append(e_local + offset)   # shift into global indexing

        # split masks
        if r in test_rounds:
            test_mask[sel] = True
        elif r in val_rounds:
            val_mask[sel] = True
        else:
            train_mask[sel] = True

        offset += len(sel)

    edge_index = (
        np.concatenate(all_edges, axis=1) if all_edges else np.empty((2, 0), dtype=np.int64)
    )

    # remove duplicate edges
    if edge_index.size:
        keys = edge_index[0] * (N + 1) + edge_index[1]
        _, uniq = np.unique(keys, return_index=True)
        edge_index = edge_index[:, np.sort(uniq)]

    # add self-loops so a node always retains its own signal (GCN convention)
    if add_self_loops:
        loops = np.stack([np.arange(N), np.arange(N)], axis=0).astype(np.int64)
        edge_index = np.concatenate([edge_index, loops], axis=1)

    return GraphData(
        x=ds.X,
        y=ds.y,
        edge_index=edge_index.astype(np.int64),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=N,
        num_features=ds.X.shape[1],
        num_classes=len(ds.class_names),
    )
