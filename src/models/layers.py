"""
Graph-neural-network layers implemented from scratch in pure PyTorch.

We deliberately avoid PyTorch-Geometric so the message-passing maths is fully
visible (and so the project has no fragile compiled dependency).  All three
layers follow the same *message -> aggregate -> update* pattern over an
``edge_index`` of shape ``(2, E)`` where ``edge_index[0]`` are source nodes and
``edge_index[1]`` are destination nodes.

* :class:`GCNLayer`      – Kipf & Welling symmetric-normalised convolution.
* :class:`GraphSAGELayer`– Hamilton et al. mean-aggregator with concat + linear.
* :class:`GATLayer`      – Veličković et al. single-head additive attention.

Aggregation is done with ``index_add_`` / ``scatter`` on the destination index,
which is exactly what a sparse spMM does but written explicitly.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Sum rows of ``src`` into ``num_nodes`` buckets given by ``index``."""
    out = src.new_zeros((num_nodes, src.size(-1)))
    out.index_add_(0, index, src)
    return out


class GCNLayer(nn.Module):
    r"""Symmetric-normalised graph convolution.

    .. math::
        H' = \sigma\!\left(\hat D^{-1/2}\hat A\,\hat D^{-1/2} H W\right)

    Implemented edge-wise: each message ``x_j`` is scaled by
    ``1/sqrt(deg_i * deg_j)`` and summed at the destination ``i``.
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        # degree of every node (edge_index already contains self-loops)
        deg = _scatter_sum(torch.ones_like(src, dtype=x.dtype).unsqueeze(-1), dst, num_nodes).squeeze(-1)
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]          # (E,)

        x = self.lin(x)
        messages = x[src] * norm.unsqueeze(-1)                # (E, out)
        return _scatter_sum(messages, dst, num_nodes)


class GraphSAGELayer(nn.Module):
    r"""GraphSAGE with a mean aggregator.

    .. math::
        h_i' = \sigma\!\big(W\,[\,h_i \,\|\, \mathrm{mean}_{j\in N(i)} h_j\,]\big)
    """

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(2 * in_dim, out_dim, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        neigh_sum = _scatter_sum(x[src], dst, num_nodes)
        deg = _scatter_sum(torch.ones_like(src, dtype=x.dtype).unsqueeze(-1), dst, num_nodes)
        neigh_mean = neigh_sum / deg.clamp(min=1)
        h = torch.cat([x, neigh_mean], dim=-1)
        return self.lin(h)


class GATLayer(nn.Module):
    r"""Single-head graph-attention layer (Veličković et al., 2018).

    Attention coefficients use the original additive form
    ``e_ij = LeakyReLU(a^T [W h_i || W h_j])`` normalised with a softmax over the
    in-neighbours of ``i``.
    """

    def __init__(self, in_dim: int, out_dim: int, negative_slope: float = 0.2):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(out_dim))
        self.att_dst = nn.Parameter(torch.empty(out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.normal_(self.att_src, std=0.1)
        nn.init.normal_(self.att_dst, std=0.1)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.size(0)

        h = self.lin(x)                                       # (N, out)
        # per-edge unnormalised attention logits
        alpha = (h[src] * self.att_src).sum(-1) + (h[dst] * self.att_dst).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)      # (E,)

        # softmax over incoming edges of each destination node
        alpha = alpha - alpha.max()                           # numerical stability
        exp = alpha.exp()
        denom = _scatter_sum(exp.unsqueeze(-1), dst, num_nodes).squeeze(-1).clamp(min=1e-16)
        norm_alpha = exp / denom[dst]                         # (E,)

        messages = h[src] * norm_alpha.unsqueeze(-1)
        out = _scatter_sum(messages, dst, num_nodes)
        return out + self.bias


LAYER_REGISTRY = {
    "gcn": GCNLayer,
    "sage": GraphSAGELayer,
    "gat": GATLayer,
}
