"""
Configurable GNN model for node-level intrusion classification.

The model is a stack of ``num_layers`` message-passing layers (GCN, GraphSAGE or
GAT — chosen by the Osprey optimiser) with ReLU activations, dropout and a final
linear classifier producing per-node logits over the five WSN-DS classes.

Every architectural knob here (``hidden_dim``, ``num_layers``, ``dropout``,
``gnn_type``) is a decision variable that the Osprey Optimization Algorithm
tunes; see :func:`config.decode_position`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import LAYER_REGISTRY


class GNNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.5, gnn_type: str = "gcn"):
        super().__init__()
        if gnn_type not in LAYER_REGISTRY:
            raise ValueError(f"unknown gnn_type {gnn_type!r}; choose {list(LAYER_REGISTRY)}")
        Layer = LAYER_REGISTRY[gnn_type]

        self.dropout = dropout
        self.gnn_type = gnn_type
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1)
        for i in range(num_layers - 1):
            self.convs.append(Layer(dims[i], dims[i + 1]))
            self.norms.append(nn.LayerNorm(dims[i + 1]))
        # final message-passing layer maps to a hidden representation,
        # then a linear head produces class logits (keeps the classifier
        # decoupled from the graph op for stability).
        last_in = dims[-1] if num_layers > 1 else in_dim
        self.convs.append(Layer(last_in, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x, edge_index).argmax(dim=-1)


def build_model(in_dim: int, num_classes: int, hp: dict) -> GNNClassifier:
    """Instantiate a :class:`GNNClassifier` from a decoded hyper-parameter dict."""
    return GNNClassifier(
        in_dim=in_dim,
        hidden_dim=hp["hidden_dim"],
        num_classes=num_classes,
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
        gnn_type=hp["gnn_type"],
    )
