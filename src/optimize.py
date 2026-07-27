"""
Osprey-driven hyper-parameter optimisation of the GNN.

This module glues the generic :class:`OspreyOptimizer` to the concrete GNN
fitness landscape:

    osprey position  --decode-->  GNN hyper-params  --train-->  val macro-F1

The Osprey algorithm maximises validation macro-F1 — or, in **resource-aware
multi-objective mode** (``OspreyConfig.efficiency_weight > 0``), the scalarised
objective

    fitness = macro_F1  -  efficiency_weight * (n_parameters / PARAM_BUDGET)

so the search is pushed toward models that are *both* accurate and small enough
for battery/CPU-constrained WSN hardware.  Each fitness evaluation trains a GNN
for a small number of epochs (``fitness_epochs``) so the search stays cheap; the
*winning* configuration is later retrained to convergence in :mod:`src.evaluate`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config import (
    DIM_NAMES,
    LOWER_BOUNDS,
    PARAM_BUDGET,
    UPPER_BOUNDS,
    OspreyConfig,
    decode_position,
)
from src.graph.builder import GraphData
from src.models.gnn import build_model
from src.osprey.optimizer import OspreyOptimizer, OspreyResult
from src.train import train_gnn


def count_parameters(in_dim: int, num_classes: int, hp: dict) -> int:
    """Number of trainable parameters of the GNN a config decodes to."""
    model = build_model(in_dim, num_classes, hp)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_fitness(graph: GraphData, fitness_epochs: int, seed: int,
                 efficiency_weight: float = 0.0):
    """Return a fitness closure evaluating a position's scalarised objective.

    With ``efficiency_weight == 0`` this is plain validation macro-F1; with a
    positive weight a normalised parameter-count penalty is subtracted, turning
    the search multi-objective (accuracy vs. deployment cost).

    A small cache avoids retraining identical decoded configurations (the OOA
    exploitation step can revisit nearby points that decode to the same
    integer/categorical hyper-parameters).
    """
    cache: dict = {}

    def fitness(position: np.ndarray) -> float:
        hp = decode_position(position)
        key = (hp["hidden_dim"], hp["num_layers"], round(hp["dropout"], 3),
               round(hp["lr"], 6), round(hp["weight_decay"], 7), hp["gnn_type"])
        if key in cache:
            return cache[key]
        outcome = train_gnn(graph, hp, epochs=fitness_epochs,
                            patience=max(8, fitness_epochs // 3), seed=seed)
        score = outcome.best_val_f1
        if efficiency_weight > 0:
            n_params = count_parameters(graph.num_features, graph.num_classes, hp)
            score -= efficiency_weight * (n_params / PARAM_BUDGET)
        cache[key] = score
        return score

    return fitness


def run_osprey_search(graph: GraphData, cfg: OspreyConfig, seed: int = 42):
    """Run OOA over the GNN search space; return (best_hp, OspreyResult)."""
    fitness = make_fitness(graph, cfg.fitness_epochs, seed,
                           efficiency_weight=cfg.efficiency_weight)
    optimizer = OspreyOptimizer(
        fitness_fn=fitness,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        n_ospreys=cfg.n_ospreys,
        n_iterations=cfg.n_iterations,
        seed=seed,
        verbose=cfg.verbose,
    )
    result: OspreyResult = optimizer.optimize()
    best_hp = decode_position(result.best_position)
    return best_hp, result
