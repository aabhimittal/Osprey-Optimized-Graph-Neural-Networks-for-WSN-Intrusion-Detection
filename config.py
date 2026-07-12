"""
Central configuration for the Osprey-Optimized GNN WSN Intrusion Detection project.

Everything that a user might reasonably want to tweak lives here so the rest of
the code stays declarative.  The most important object is ``SEARCH_SPACE`` — it
defines the *decision variables* that the Osprey Optimization Algorithm (OOA)
tunes.  Each dimension of the osprey's position vector maps to one GNN
hyper-parameter through :func:`decode_position`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Attack taxonomy (WSN-DS: a LEACH-based WSN intrusion-detection dataset)
# --------------------------------------------------------------------------- #
# 0 = Normal traffic, 1..4 = the four canonical routing-layer attacks that the
# WSN-DS benchmark models.  The synthetic generator reproduces the behavioural
# signature of each of these classes.
CLASS_NAMES: List[str] = ["Normal", "Blackhole", "Grayhole", "Flooding", "Scheduling"]
NUM_CLASSES: int = len(CLASS_NAMES)

# The 18 behavioural features recorded per node per LEACH round in WSN-DS.
FEATURE_NAMES: List[str] = [
    "Time",             # simulation time of the round
    "Is_CH",            # 1 if the node is a Cluster Head this round
    "Dist_To_CH",       # distance from the node to its Cluster Head
    "ADV_S",            # advertisement (CH broadcast) messages sent
    "ADV_R",            # advertisement messages received
    "JOIN_S",           # join-request messages sent
    "JOIN_R",           # join-request messages received
    "SCH_S",            # TDMA schedule messages sent
    "SCH_R",            # TDMA schedule messages received
    "Rank",             # node's TDMA slot / rank inside its cluster
    "DATA_S",           # data packets sent
    "DATA_R",           # data packets received (relayed) by the node
    "Data_Sent_To_BS",  # data packets actually forwarded to the Base Station
    "Dist_CH_To_BS",    # distance from the CH to the Base Station
    "Send_Code",        # cluster send code
    "Consumed_Energy",  # energy expended during the round
    "Rank_Ratio",       # engineered: rank relative to cluster size
    "Forward_Ratio",    # engineered: Data_Sent_To_BS / (DATA_R + 1)  -> exposes black/grayholes
]
NUM_FEATURES: int = len(FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Dataset generation defaults
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    n_rounds: int = 60           # number of LEACH rounds -> each round is one graph
    n_nodes: int = 80            # sensor nodes per round
    ch_fraction: float = 0.05    # ~5 % of nodes become Cluster Heads (LEACH default p)
    attack_fraction: float = 0.30  # fraction of node-rounds that are malicious
    field_size: float = 200.0    # deployment area is field_size x field_size metres
    knn_k: int = 6               # k for the spatial k-NN edges added on top of the tree
    label_noise: float = 0.03    # fraction of labels randomly flipped (realism)
    csv_path: str = "data/wsn_ds_synthetic.csv"


# --------------------------------------------------------------------------- #
# Osprey Optimization Algorithm search space
# --------------------------------------------------------------------------- #
# Each entry is a continuous interval [low, high].  OOA searches this box; the
# ``decode_position`` function below turns a continuous position into a concrete
# (typed) GNN hyper-parameter dictionary.  This decoupling lets a *continuous*
# metaheuristic optimise a *mixed* (integer/categorical/continuous) space.
SEARCH_SPACE = {
    "hidden_dim":   (16.0, 128.0),   # -> rounded to nearest multiple of 8
    "num_layers":   (2.0, 4.0),      # -> rounded to int
    "dropout":      (0.0, 0.6),      # continuous
    "log_lr":       (-4.0, -2.0),    # learning rate = 10**log_lr  (1e-4 .. 1e-2)
    "log_wd":       (-6.0, -3.0),    # weight decay  = 10**log_wd  (1e-6 .. 1e-3)
    "gnn_type":     (0.0, 2.999),    # -> int index into GNN_TYPES
}
GNN_TYPES: List[str] = ["gcn", "sage", "gat"]

# convenient array views
DIM_NAMES: List[str] = list(SEARCH_SPACE.keys())
LOWER_BOUNDS: np.ndarray = np.array([v[0] for v in SEARCH_SPACE.values()], dtype=float)
UPPER_BOUNDS: np.ndarray = np.array([v[1] for v in SEARCH_SPACE.values()], dtype=float)
DIM: int = len(SEARCH_SPACE)


def decode_position(position: np.ndarray) -> dict:
    """Map a continuous OOA position vector to a typed GNN hyper-parameter dict.

    Parameters
    ----------
    position : np.ndarray, shape (DIM,)
        A point inside the search-space box (values are clipped to the bounds
        first, so callers need not pre-clip).

    Returns
    -------
    dict
        ``hidden_dim`` (int), ``num_layers`` (int), ``dropout`` (float),
        ``lr`` (float), ``weight_decay`` (float), ``gnn_type`` (str).
    """
    p = np.clip(np.asarray(position, dtype=float), LOWER_BOUNDS, UPPER_BOUNDS)
    idx = {name: i for i, name in enumerate(DIM_NAMES)}

    hidden = int(round(p[idx["hidden_dim"]] / 8.0)) * 8      # multiples of 8
    hidden = max(8, hidden)
    return {
        "hidden_dim": hidden,
        "num_layers": int(round(p[idx["num_layers"]])),
        "dropout": float(p[idx["dropout"]]),
        "lr": float(10.0 ** p[idx["log_lr"]]),
        "weight_decay": float(10.0 ** p[idx["log_wd"]]),
        "gnn_type": GNN_TYPES[int(p[idx["gnn_type"]])],
    }


# --------------------------------------------------------------------------- #
# Osprey optimiser defaults
# --------------------------------------------------------------------------- #
@dataclass
class OspreyConfig:
    n_ospreys: int = 6       # population size
    n_iterations: int = 8    # generations
    # Fitness = validation macro-F1 of a GNN trained with the decoded config.
    fitness_epochs: int = 25  # short training budget used *inside* the search
    verbose: bool = True


# --------------------------------------------------------------------------- #
# Final-model training defaults (used once the best config is found)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    epochs: int = 150
    patience: int = 25       # early-stopping patience on validation macro-F1
    val_fraction: float = 0.2
    test_fraction: float = 0.2
    device: str = "cpu"


# --------------------------------------------------------------------------- #
# "quick" preset — tiny budgets so the whole pipeline runs in ~1-2 min on CPU.
# Enabled from the CLI with `--quick`.
# --------------------------------------------------------------------------- #
def apply_quick(data: DataConfig, osp: OspreyConfig, tr: TrainConfig) -> None:
    data.n_rounds = 40
    data.n_nodes = 60
    osp.n_ospreys = 5
    osp.n_iterations = 5
    osp.fitness_epochs = 20
    tr.epochs = 60
    tr.patience = 15
