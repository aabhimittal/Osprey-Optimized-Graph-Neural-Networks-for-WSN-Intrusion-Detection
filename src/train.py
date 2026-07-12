"""
Training / evaluation of a single GNN configuration.

:func:`train_gnn` performs full-batch node-classification training on the
disjoint-union graph with early stopping on validation macro-F1.  It is used in
two places:

* inside the Osprey search as the **fitness function** (short budget), and
* for the **final model** once the best config is known (long budget).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import NUM_CLASSES
from src.graph.builder import GraphData
from src.models.gnn import build_model
from src.utils import classification_metrics


@dataclass
class TrainOutcome:
    model: nn.Module
    best_val_f1: float
    test_metrics: dict
    val_metrics: dict
    train_history: list
    val_history: list
    epochs_run: int
    y_test_true: np.ndarray
    y_test_pred: np.ndarray


def _class_weights(y: np.ndarray, mask: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights (attacks are rarer than normal traffic)."""
    counts = np.bincount(y[mask], minlength=num_classes).astype(float)
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_gnn(graph: GraphData, hp: dict, epochs: int = 100, patience: int = 20,
              device: str = "cpu", seed: int = 42,
              verbose: bool = False) -> TrainOutcome:
    """Train a GNN with hyper-parameters ``hp`` and return the outcome."""
    torch.manual_seed(seed)

    x = torch.tensor(graph.x, dtype=torch.float32, device=device)
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    y = torch.tensor(graph.y, dtype=torch.long, device=device)
    train_mask = torch.tensor(graph.train_mask, device=device)
    val_mask = torch.tensor(graph.val_mask, device=device)
    test_mask = torch.tensor(graph.test_mask, device=device)

    model = build_model(graph.num_features, graph.num_classes, hp).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"],
                                 weight_decay=hp["weight_decay"])
    weights = _class_weights(graph.y, graph.train_mask, graph.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    train_history, val_history = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # --- validation ---
        model.eval()
        with torch.no_grad():
            val_pred = model(x, edge_index)[val_mask].argmax(-1).cpu().numpy()
        val_true = graph.y[graph.val_mask]
        val_f1 = classification_metrics(val_true, val_pred, graph.num_classes)["macro_f1"]

        train_history.append(float(loss.item()))
        val_history.append(float(val_f1))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d} | loss {loss.item():.4f} | val macro-F1 {val_f1:.4f}")

        if epochs_no_improve >= patience:
            break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- final evaluation on val + test ---
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        val_pred = out[val_mask].argmax(-1).cpu().numpy()
        test_pred = out[test_mask].argmax(-1).cpu().numpy()

    val_metrics = classification_metrics(graph.y[graph.val_mask], val_pred, graph.num_classes)
    test_metrics = classification_metrics(graph.y[graph.test_mask], test_pred, graph.num_classes)

    return TrainOutcome(
        model=model,
        best_val_f1=float(best_val_f1),
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        train_history=train_history,
        val_history=val_history,
        epochs_run=epoch,
        y_test_true=graph.y[graph.test_mask],
        y_test_pred=test_pred,
    )
