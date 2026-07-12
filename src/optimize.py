"""
Osprey-driven hyper-parameter optimisation of the GNN.

This module glues the generic :class:`OspreyOptimizer` to the concrete GNN
fitness landscape:

    osprey position  --decode-->  GNN hyper-params  --train-->  val macro-F1

The Osprey algorithm maximises validation macro-F1.  Each fitness evaluation
trains a GNN for a small number of epochs (``fitness_epochs``) so the search
stays cheap; the *winning* configuration is later retrained to convergence in
:mod:`src.evaluate`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from config import (
    DIM_NAMES,
    LOWER_BOUNDS,
    UPPER_BOUNDS,
    OspreyConfig,
    decode_position,
)
from src.graph.builder import GraphData
from src.osprey.optimizer import OspreyOptimizer, OspreyResult
from src.train import train_gnn


def make_fitness(graph: GraphData, fitness_epochs: int, seed: int):
    """Return a fitness closure evaluating a position's validation macro-F1.

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
        cache[key] = outcome.best_val_f1
        return outcome.best_val_f1

    return fitness


def run_osprey_search(graph: GraphData, cfg: OspreyConfig, seed: int = 42):
    """Run OOA over the GNN search space; return (best_hp, OspreyResult)."""
    fitness = make_fitness(graph, cfg.fitness_epochs, seed)
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
