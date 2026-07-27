#!/usr/bin/env python3
"""
End-to-end pipeline: Osprey-Optimized GNN for WSN Intrusion Detection.

Stages (each is an independent CLI flag; ``--all`` runs the whole thing):

    1. --generate    build the synthetic WSN-DS dataset (skips if real WSN-DS.csv present)
    2. --optimize    run the Osprey Optimization Algorithm to tune the GNN
    3. --evaluate    retrain the best GNN, compare against baselines, write results/
    4. --explain     saliency + neighbour-occlusion explanations for the trained model
    5. --trust       cross-round trust/reputation scoring of every physical node
    6. --robustness  macro-F1 under feature noise / evasion, GNN vs RandomForest

Extra switches:
    --multi-objective   make the Osprey fitness resource-aware
                        (macro-F1 minus a model-size penalty)
    --quick             tiny budgets for a fast smoke run

Typical usage
-------------
    python main.py --all              # full run incl. explain/trust/robustness
    python main.py --all --quick      # tiny budgets, ~1-2 min, for a fast demo
    python main.py --optimize --multi-objective
    python main.py --explain --trust --robustness   # post-hoc analyses only
"""
from __future__ import annotations

import argparse
import os
import sys

# make "config" and "src" importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (  # noqa: E402
    DataConfig,
    OspreyConfig,
    TrainConfig,
    apply_quick,
)
from src.utils import save_json, set_seed  # noqa: E402

SEED = 42
BEST_CONFIG_PATH = "results/best_config.json"
SUMMARY_PATH = "results/metrics.json"


def stage_generate(data_cfg: DataConfig) -> None:
    from src.data.synthetic import generate_and_save

    if os.path.exists("data/WSN-DS.csv"):
        print("[generate] real data/WSN-DS.csv detected — the loader will use it; "
              "skipping synthetic generation.")
        return
    print(f"[generate] creating synthetic WSN-DS: {data_cfg.n_rounds} rounds x "
          f"{data_cfg.n_nodes} nodes -> {data_cfg.csv_path}")
    df = generate_and_save(data_cfg, SEED)
    dist = df["label"].value_counts().sort_index()
    from config import CLASS_NAMES

    print("[generate] class distribution:")
    for lbl, cnt in dist.items():
        print(f"           {CLASS_NAMES[int(lbl)]:<12} {cnt}")


def _load_graph(data_cfg: DataConfig, tr_cfg: TrainConfig, return_dataset: bool = False):
    from src.data.loader import load_dataset
    from src.graph.builder import build_graph

    csv = "data/WSN-DS.csv" if os.path.exists("data/WSN-DS.csv") else data_cfg.csv_path
    ds = load_dataset(csv, seed=SEED)
    graph = build_graph(ds, knn_k=data_cfg.knn_k,
                        val_fraction=tr_cfg.val_fraction,
                        test_fraction=tr_cfg.test_fraction, seed=SEED)
    print(f"[graph] {graph.num_nodes} nodes | {graph.edge_index.shape[1]} edges | "
          f"{graph.num_features} features | {graph.num_classes} classes")
    print(f"[graph] split: train={int(graph.train_mask.sum())} "
          f"val={int(graph.val_mask.sum())} test={int(graph.test_mask.sum())}")
    return (graph, ds) if return_dataset else graph


def _load_checkpoint(path: str = "results/best_model.pt"):
    """Rebuild the trained GNN from the saved checkpoint."""
    import torch

    from src.models.gnn import build_model

    if not os.path.exists(path):
        print(f"[load] no checkpoint at {path}; run --evaluate first.")
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["feature_dim"], ckpt["num_classes"], ckpt["hyperparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def stage_optimize(data_cfg: DataConfig, osp_cfg: OspreyConfig, tr_cfg: TrainConfig):
    from src.optimize import count_parameters, run_osprey_search

    graph = _load_graph(data_cfg, tr_cfg)
    print(f"[optimize] running Osprey: {osp_cfg.n_ospreys} ospreys x "
          f"{osp_cfg.n_iterations} iterations ...")
    best_hp, result = run_osprey_search(graph, osp_cfg, seed=SEED)
    n_params = count_parameters(graph.num_features, graph.num_classes, best_hp)
    print(f"[optimize] best fitness = {result.best_fitness:.4f} "
          f"(efficiency_weight={osp_cfg.efficiency_weight})")
    print(f"[optimize] best hyper-parameters: {best_hp} | {n_params:,} parameters")
    save_json(
        {"best_hyperparameters": best_hp,
         "best_fitness": result.best_fitness,
         "n_parameters": n_params,
         "efficiency_weight": osp_cfg.efficiency_weight,
         "convergence_history": result.history,
         "population_mean_history": result.mean_history,
         "total_evaluations": result.evaluations},
        BEST_CONFIG_PATH,
    )
    print(f"[optimize] saved -> {BEST_CONFIG_PATH}")
    return graph, best_hp, result


def stage_evaluate(graph, best_hp, result, tr_cfg: TrainConfig):
    from src.evaluate import run_final_evaluation

    summary = run_final_evaluation(graph, best_hp, tr_cfg, osprey_result=result, seed=SEED)
    save_json(summary, SUMMARY_PATH)

    print("\n================ RESULTS (test set) ================")
    for name, m in summary["results"].items():
        print(f"  {name:<22} | acc {m['accuracy']:.3f} | macro-F1 {m['macro_f1']:.3f} "
              f"| macro-Recall {m['macro_recall']:.3f}")
    print("====================================================")
    print(f"[evaluate] wrote {SUMMARY_PATH} and plots to results/")
    return summary


def stage_explain(model, graph):
    from src.explain import run_explain

    print("[explain] computing gradient saliency + neighbour occlusion ...")
    report = run_explain(model, graph)
    print("[explain] top features per class:")
    for cls, feats in report["top_features_per_class"].items():
        print(f"           {cls:<12} {', '.join(feats)}")
    print("[explain] wrote results/explainability.png and results/explanations.json")


def stage_trust(model, graph, ds):
    from src.trust import run_trust

    print("[trust] scoring per-node trust across rounds ...")
    report = run_trust(model, graph, ds)
    print(f"[trust] persistent attackers: {report['n_compromised']} | "
          f"flagged: {report['n_flagged']} | "
          f"precision {report['precision']:.3f} | recall {report['recall']:.3f} | "
          f"F1 {report['f1']:.3f}")
    if report["mean_detection_latency_rounds"] is not None:
        print(f"[trust] mean detection latency: "
              f"{report['mean_detection_latency_rounds']:.1f} rounds")
    sr = report["single_round_reference"]
    print(f"[trust] single-round reference F1: {sr['f1']:.3f}  "
          f"(trust-based F1: {report['f1']:.3f})")
    print("[trust] wrote results/trust_scores.png and results/trust_report.json")


def stage_robustness(model, graph):
    from src.robustness import run_robustness

    print("[robustness] sweeping feature perturbations (noise + evasion) ...")
    report = run_robustness(model, graph, seed=SEED)
    eps = report["epsilons"]
    print(f"[robustness] eps={eps[-1]:.1f}: GNN evasion F1 "
          f"{report['gnn_evasion'][-1]:.3f} vs RF evasion F1 {report['rf_evasion'][-1]:.3f}")
    print("[robustness] wrote results/robustness.png and results/robustness.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true", help="build the dataset")
    ap.add_argument("--optimize", action="store_true", help="run the Osprey search")
    ap.add_argument("--evaluate", action="store_true", help="final train + baselines")
    ap.add_argument("--explain", action="store_true",
                    help="saliency + neighbour-occlusion explanations")
    ap.add_argument("--trust", action="store_true",
                    help="cross-round trust/reputation scoring")
    ap.add_argument("--robustness", action="store_true",
                    help="feature-perturbation robustness sweep")
    ap.add_argument("--all", action="store_true",
                    help="run every stage: generate → optimize → evaluate → "
                         "explain → trust → robustness")
    ap.add_argument("--multi-objective", action="store_true", dest="multi_objective",
                    help="resource-aware Osprey fitness (macro-F1 minus a "
                         "normalised model-size penalty)")
    ap.add_argument("--quick", action="store_true", help="tiny budgets for a fast demo")
    args = ap.parse_args()

    stage_flags = [args.generate, args.optimize, args.evaluate,
                   args.explain, args.trust, args.robustness, args.all]
    if not any(stage_flags):
        ap.print_help()
        return

    set_seed(SEED)
    data_cfg, osp_cfg, tr_cfg = DataConfig(), OspreyConfig(), TrainConfig()
    if args.quick:
        apply_quick(data_cfg, osp_cfg, tr_cfg)
        print("[config] QUICK mode enabled (small budgets).")
    if args.multi_objective:
        osp_cfg.efficiency_weight = 0.15
        print(f"[config] multi-objective search: fitness = macro-F1 - "
              f"{osp_cfg.efficiency_weight} * (params / budget)")

    do_gen = args.generate or args.all
    do_opt = args.optimize or args.all
    do_eval = args.evaluate or args.all
    do_explain = args.explain or args.all
    do_trust = args.trust or args.all
    do_rob = args.robustness or args.all

    if do_gen:
        stage_generate(data_cfg)

    graph = ds = best_hp = result = None
    if do_opt:
        graph, best_hp, result = stage_optimize(data_cfg, osp_cfg, tr_cfg)

    if do_eval:
        if best_hp is None:
            # load a previously-saved best config
            import json

            if not os.path.exists(BEST_CONFIG_PATH):
                print(f"[evaluate] no {BEST_CONFIG_PATH}; run --optimize first.")
                return
            with open(BEST_CONFIG_PATH) as fh:
                best_hp = json.load(fh)["best_hyperparameters"]
            graph = _load_graph(data_cfg, tr_cfg)
        stage_evaluate(graph, best_hp, result, tr_cfg)

    # ---- post-hoc analyses on the trained checkpoint ----
    if do_explain or do_trust or do_rob:
        model = _load_checkpoint()
        if model is None:
            return
        if graph is None or ds is None:
            graph, ds = _load_graph(data_cfg, tr_cfg, return_dataset=True)
        if do_explain:
            stage_explain(model, graph)
        if do_trust:
            stage_trust(model, graph, ds)
        if do_rob:
            stage_robustness(model, graph)


if __name__ == "__main__":
    main()
