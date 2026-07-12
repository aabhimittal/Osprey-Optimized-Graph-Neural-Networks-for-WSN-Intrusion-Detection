"""Tests for the from-scratch GNN layers and the classifier."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import decode_position, LOWER_BOUNDS, UPPER_BOUNDS
from src.models.gnn import GNNClassifier, build_model
from src.models.layers import GATLayer, GCNLayer, GraphSAGELayer


def _toy_graph():
    # 4 nodes in a path 0-1-2-3 plus self loops, 3 features
    x = torch.randn(4, 3)
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    edges += [(i, i) for i in range(4)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


def test_layers_output_shapes():
    x, ei = _toy_graph()
    for Layer in (GCNLayer, GraphSAGELayer, GATLayer):
        layer = Layer(3, 7)
        out = layer(x, ei)
        assert out.shape == (4, 7)
        assert torch.isfinite(out).all()


def test_gat_attention_normalises():
    # with only self-loops, every node attends to itself with weight 1,
    # so GAT output should equal W x (up to bias).
    x = torch.randn(5, 4)
    ei = torch.tensor([[i for i in range(5)], [i for i in range(5)]], dtype=torch.long)
    layer = GATLayer(4, 4)
    out = layer(x, ei)
    expected = layer.lin(x) + layer.bias
    assert torch.allclose(out, expected, atol=1e-5)


def test_classifier_forward_all_types():
    x, ei = _toy_graph()
    for gnn in ("gcn", "sage", "gat"):
        model = GNNClassifier(3, 8, num_classes=5, num_layers=3, dropout=0.1, gnn_type=gnn)
        logits = model(x, ei)
        assert logits.shape == (4, 5)
        preds = model.predict(x, ei)
        assert preds.shape == (4,)
        assert preds.max() < 5 and preds.min() >= 0


def test_build_model_from_decoded_position():
    mid = (LOWER_BOUNDS + UPPER_BOUNDS) / 2
    hp = decode_position(mid)
    model = build_model(3, 5, hp)
    x, ei = _toy_graph()
    assert model(x, ei).shape == (4, 5)


def test_gradients_flow():
    x, ei = _toy_graph()
    y = torch.tensor([0, 1, 2, 3])
    model = GNNClassifier(3, 16, num_classes=5, num_layers=2, gnn_type="sage")
    logits = model(x, ei)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
