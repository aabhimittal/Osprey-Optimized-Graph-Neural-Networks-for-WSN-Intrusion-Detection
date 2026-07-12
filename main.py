#!/usr/bin/env python3
"""
End-to-end pipeline: Osprey-Optimized GNN for WSN Intrusion Detection.

Stages (each is an independent CLI flag; ``--all`` runs the whole thing):

    1. --generate   build the synthetic WSN-DS dataset (skips if real WSN-DS.csv present)
    2. --optimize   run the Osprey Optimization Algorithm to tune the GNN
    3. --evaluate   retrain the best GNN, compare against baselines, write results/

Typical usage
-------------
    python main.py --all              # full run (~5-10 min on CPU)
    python main.py --all --quick      # tiny budgets, ~1-2 min, for a fast demo
    python main.py --generate         # just (re)build the dataset
    python main.py --optimize --evaluate
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


def _load_graph(data_cfg: DataConfig, tr_cfg: TrainConfig):
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
    return graph


def stage_optimize(data_cfg: DataConfig, osp_cfg: OspreyConfig, tr_cfg: TrainConfig):
    from src.optimize import run_osprey_search

    graph = _load_graph(data_cfg, tr_cfg)
    print(f"[optimize] running Osprey: {osp_cfg.n_ospreys} ospreys x "
          f"{osp_cfg.n_iterations} iterations ...")
    best_hp, result = run_osprey_search(graph, osp_cfg, seed=SEED)
    print(f"[optimize] best validation macro-F1 = {result.best_fitness:.4f}")
    print(f"[optimize] best hyper-parameters: {best_hp}")
    save_json(
        {"best_hyperparameters": best_hp,
         "best_val_macro_f1": result.best_fitness,
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generate", action="store_true", help="build the dataset")
    ap.add_argument("--optimize", action="store_true", help="run the Osprey search")
    ap.add_argument("--evaluate", action="store_true", help="final train + baselines")
    ap.add_argument("--all", action="store_true", help="run generate + optimize + evaluate")
    ap.add_argument("--quick", action="store_true", help="tiny budgets for a fast demo")
    args = ap.parse_args()

    if not any([args.generate, args.optimize, args.evaluate, args.all]):
        ap.print_help()
        return

    set_seed(SEED)
    data_cfg, osp_cfg, tr_cfg = DataConfig(), OspreyConfig(), TrainConfig()
    if args.quick:
        apply_quick(data_cfg, osp_cfg, tr_cfg)
        print("[config] QUICK mode enabled (small budgets).")

    do_gen = args.generate or args.all
    do_opt = args.optimize or args.all
    do_eval = args.evaluate or args.all

    if do_gen:
        stage_generate(data_cfg)

    graph = best_hp = result = None
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


if __name__ == "__main__":
    main()
