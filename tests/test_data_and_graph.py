"""Tests for the synthetic generator, loader and graph builder."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_NAMES, FEATURE_NAMES, DataConfig
from src.data.loader import Dataset, load_dataset
from src.data.synthetic import generate
from src.graph.builder import build_graph


def _small_cfg(tmp_path):
    cfg = DataConfig()
    cfg.n_rounds = 8
    cfg.n_nodes = 40
    cfg.csv_path = os.path.join(tmp_path, "wsn.csv")
    return cfg


def test_generator_schema_and_labels(tmp_path):
    cfg = _small_cfg(tmp_path)
    df = generate(cfg, seed=0)
    # every declared feature exists
    for f in FEATURE_NAMES:
        assert f in df.columns
    # labels are within range
    assert df["label"].between(0, len(CLASS_NAMES) - 1).all()
    # all classes appear (with 30% attacks over many nodes this is safe)
    assert df["label"].nunique() == len(CLASS_NAMES)
    # blackholes really do drop data: low forward ratio on average
    bh = df[df["label"] == 1]
    normal_ch = df[(df["label"] == 0) & (df["Is_CH"] == 1)]
    assert bh["Forward_Ratio"].mean() < normal_ch["Forward_Ratio"].mean()


def test_loader_shapes(tmp_path):
    cfg = _small_cfg(tmp_path)
    df = generate(cfg, seed=0)
    df.to_csv(cfg.csv_path, index=False)
    ds = load_dataset(cfg.csv_path, seed=0)
    assert isinstance(ds, Dataset)
    assert ds.X.shape[0] == len(df)
    assert ds.X.shape[1] == len(FEATURE_NAMES)
    assert ds.y.shape[0] == ds.X.shape[0]
    # features are standard-scaled -> roughly zero mean
    assert abs(ds.X.mean()) < 0.5


def test_graph_builder_shapes_and_split(tmp_path):
    cfg = _small_cfg(tmp_path)
    df = generate(cfg, seed=0)
    df.to_csv(cfg.csv_path, index=False)
    ds = load_dataset(cfg.csv_path, seed=0)
    g = build_graph(ds, knn_k=5, seed=0)

    assert g.edge_index.shape[0] == 2
    assert g.edge_index.max() < g.num_nodes
    assert g.edge_index.min() >= 0
    # masks partition the nodes exactly once
    total = g.train_mask.astype(int) + g.val_mask.astype(int) + g.test_mask.astype(int)
    assert np.all(total == 1)
    # self-loops present -> every node is its own neighbour somewhere
    self_loops = (g.edge_index[0] == g.edge_index[1]).sum()
    assert self_loops == g.num_nodes


def test_split_is_by_round_no_leakage(tmp_path):
    cfg = _small_cfg(tmp_path)
    df = generate(cfg, seed=0)
    df.to_csv(cfg.csv_path, index=False)
    ds = load_dataset(cfg.csv_path, seed=0)
    g = build_graph(ds, knn_k=5, seed=0)
    # a round must belong to exactly one split
    for r in np.unique(ds.round_id):
        sel = ds.round_id == r
        in_train = g.train_mask[sel].any()
        in_val = g.val_mask[sel].any()
        in_test = g.test_mask[sel].any()
        assert sum([in_train, in_val, in_test]) == 1
