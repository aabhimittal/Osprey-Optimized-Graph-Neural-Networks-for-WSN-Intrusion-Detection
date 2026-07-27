"""Tests for the novel-feature modules: persistence, multi-objective fitness,
explainability, trust scoring and robustness."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_NAMES, DataConfig, PARAM_BUDGET
from src.data.loader import load_dataset
from src.data.synthetic import generate
from src.graph.builder import build_graph
from src.models.gnn import build_model
from src.optimize import count_parameters
from src.train import train_gnn

HP = {"hidden_dim": 16, "num_layers": 2, "dropout": 0.1,
      "lr": 5e-3, "weight_decay": 1e-4, "gnn_type": "sage"}


def _pipeline(tmp_path, n_rounds=10, n_nodes=40):
    cfg = DataConfig()
    cfg.n_rounds = n_rounds
    cfg.n_nodes = n_nodes
    cfg.csv_path = os.path.join(tmp_path, "wsn.csv")
    df = generate(cfg, seed=0)
    df.to_csv(cfg.csv_path, index=False)
    ds = load_dataset(cfg.csv_path, seed=0)
    graph = build_graph(ds, knn_k=5, seed=0)
    outcome = train_gnn(graph, HP, epochs=30, patience=30, seed=0)
    return cfg, df, ds, graph, outcome.model


# --------------------------------------------------------------------------- #
# persistent attackers in the generator
# --------------------------------------------------------------------------- #
def test_generator_has_persistent_attackers(tmp_path):
    cfg = DataConfig()
    cfg.n_rounds = 12
    cfg.n_nodes = 50
    df = generate(cfg, seed=1)
    # a persistent attacker is malicious in (almost) every round; with 60 % of a
    # 30 % budget persistent there must be nodes malicious in >= 80 % of rounds
    frac = df[df["label"] > 0].groupby("node_id")["round"].nunique() / cfg.n_rounds
    assert (frac >= 0.8).sum() >= 3, "expected several persistent attackers"
    # ... and they keep a consistent attack type (ignoring label noise)
    nid = frac.idxmax()
    types = df[(df["node_id"] == nid) & (df["label"] > 0)]["label"].value_counts()
    assert types.iloc[0] / types.sum() > 0.8


def test_generator_no_persistence_when_disabled(tmp_path):
    cfg = DataConfig()
    cfg.n_rounds = 12
    cfg.n_nodes = 50
    cfg.persistent_attacker_fraction = 0.0
    cfg.label_noise = 0.0
    df = generate(cfg, seed=1)
    frac = df[df["label"] > 0].groupby("node_id")["round"].nunique() / cfg.n_rounds
    # with purely random per-round attackers, being malicious in >=90 % of 12
    # rounds is (0.3)^11-level unlikely
    assert (frac >= 0.9).sum() == 0


# --------------------------------------------------------------------------- #
# multi-objective fitness
# --------------------------------------------------------------------------- #
def test_count_parameters_matches_manual():
    hp = dict(HP)
    n = count_parameters(18, 5, hp)
    model = build_model(18, 5, hp)
    assert n == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 0 < n < PARAM_BUDGET


def test_efficiency_weight_penalises_big_models(tmp_path):
    from src.optimize import make_fitness
    from config import DIM_NAMES

    cfg, df, ds, graph, _ = _pipeline(tmp_path, n_rounds=6, n_nodes=30)
    idx = {n: i for i, n in enumerate(DIM_NAMES)}
    small = np.array([16, 2, 0.1, -2.3, -4.0, 1.5])   # hidden 16, sage
    big = small.copy()
    big[idx["hidden_dim"]] = 128                       # hidden 128, same otherwise

    f_plain = make_fitness(graph, fitness_epochs=5, seed=0, efficiency_weight=0.0)
    f_multi = make_fitness(graph, fitness_epochs=5, seed=0, efficiency_weight=0.5)
    # the penalty must strictly lower the big model's fitness relative to plain
    assert f_multi(big) < f_plain(big)
    # and the gap must exceed the small model's own penalty gap
    gap_small = f_plain(small) - f_multi(small)
    gap_big = f_plain(big) - f_multi(big)
    assert gap_big > gap_small


# --------------------------------------------------------------------------- #
# explainability
# --------------------------------------------------------------------------- #
def test_feature_saliency_shapes_and_distribution(tmp_path):
    from src.explain import feature_saliency

    cfg, df, ds, graph, model = _pipeline(tmp_path)
    sal = feature_saliency(model, graph, mask=graph.test_mask)
    assert sal.shape == (graph.num_classes, graph.num_features)
    assert np.all(sal >= 0)
    # each non-empty class row is a probability distribution
    for c in range(graph.num_classes):
        s = sal[c].sum()
        assert s == 0 or abs(s - 1.0) < 1e-6


def test_neighbour_occlusion_ranks_neighbours(tmp_path):
    from src.explain import neighbour_occlusion

    cfg, df, ds, graph, model = _pipeline(tmp_path)
    node = int(np.where(graph.test_mask)[0][0])
    ranking = neighbour_occlusion(model, graph, node)
    assert len(ranking) > 0
    drops = [r["logit_drop"] for r in ranking]
    assert drops == sorted(drops, reverse=True)


# --------------------------------------------------------------------------- #
# trust scoring
# --------------------------------------------------------------------------- #
def test_trust_ema_bounds_and_separation(tmp_path):
    from src.trust import compute_trust

    cfg, df, ds, graph, model = _pipeline(tmp_path, n_rounds=12, n_nodes=50)
    tr = compute_trust(model, graph, ds, alpha=0.7)
    assert tr["trust"].shape == (len(tr["node_ids"]), len(tr["rounds"]))
    assert np.all(tr["trust"] >= 0) and np.all(tr["trust"] <= 1)
    # persistent attackers should end with lower trust than honest nodes
    compromised = tr["mal_fraction"] >= 0.8
    honest = tr["mal_fraction"] <= 0.05
    if compromised.any() and honest.any():
        assert tr["trust"][compromised, -1].mean() < tr["trust"][honest, -1].mean()


def test_evaluate_trust_report_fields(tmp_path):
    from src.trust import compute_trust, evaluate_trust

    cfg, df, ds, graph, model = _pipeline(tmp_path, n_rounds=12, n_nodes=50)
    tr = compute_trust(model, graph, ds)
    rep = evaluate_trust(tr)
    for key in ("precision", "recall", "f1", "n_compromised", "n_flagged"):
        assert key in rep
    assert 0 <= rep["precision"] <= 1 and 0 <= rep["recall"] <= 1


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def test_robustness_report(tmp_path):
    from src.robustness import EPSILONS, run_robustness

    cfg, df, ds, graph, model = _pipeline(tmp_path)
    rep = run_robustness(model, graph, results_dir=str(tmp_path), seed=0)
    for key in ("gnn_noise", "rf_noise", "gnn_evasion", "rf_evasion"):
        assert len(rep[key]) == len(EPSILONS)
        assert all(0 <= v <= 1 for v in rep[key])
    # eps = 0 must equal the clean score in both scenarios
    assert abs(rep["gnn_noise"][0] - rep["gnn_evasion"][0]) < 1e-9
    assert os.path.exists(os.path.join(str(tmp_path), "robustness.png"))
