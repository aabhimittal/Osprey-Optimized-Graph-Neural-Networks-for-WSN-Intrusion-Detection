# System Architecture

## End-to-end pipeline

```mermaid
flowchart TD
    A["WSN-DS data<br/>(synthetic generator<br/>or real CSV)"] --> B["Preprocessing<br/>clean + standard-scale<br/>src/data/loader.py"]
    B --> C["Graph construction<br/>cluster + CH-backbone + kNN edges<br/>disjoint-union graph<br/>src/graph/builder.py"]
    C --> D{"Osprey Optimization<br/>src/osprey + src/optimize.py"}
    D -->|"decode position → GNN config"| E["Train GNN (short)<br/>src/train.py"]
    E -->|"validation macro-F1 = fitness"| D
    D -->|"best hyper-parameters"| F["Final training<br/>GNN to convergence<br/>src/train.py"]
    F --> G["Evaluation + baselines<br/>RF · MLP · untuned GNN<br/>src/evaluate.py"]
    G --> H["results/<br/>metrics.json · plots · best_model.pt"]
```

## The optimisation loop

```mermaid
flowchart LR
    P["Osprey population<br/>(hyper-param vectors)"] --> Q["Phase 1: exploration<br/>dive toward a better 'fish'"]
    Q --> R["Phase 2: exploitation<br/>carry fish (1/t step)"]
    R --> S["Greedy selection<br/>keep if fitness ↑"]
    S --> T{"t = T ?"}
    T -->|no| Q
    T -->|yes| U["best GNN config"]
```

## GNN forward pass (one layer)

```mermaid
flowchart LR
    X["node features H"] --> M["message: transform<br/>neighbour features"]
    M --> AGG["aggregate at destination<br/>(index_add over edges)"]
    AGG --> UPD["update: LayerNorm → ReLU → dropout"]
    UPD --> Y["H'"]
```

## Repository layout

```
.
├── main.py                     # CLI: --generate / --optimize / --evaluate / --all / --quick
├── config.py                   # all knobs; SEARCH_SPACE + decode_position
├── src/
│   ├── data/
│   │   ├── synthetic.py        # WSN-DS-schema synthetic generator
│   │   └── loader.py           # load/scale synthetic OR real WSN-DS
│   ├── graph/
│   │   └── builder.py          # per-round graphs → disjoint-union + split masks
│   ├── models/
│   │   ├── layers.py           # GCN / GraphSAGE / GAT from scratch
│   │   └── gnn.py              # configurable GNNClassifier
│   ├── osprey/
│   │   └── optimizer.py        # Osprey Optimization Algorithm
│   ├── train.py                # train/eval one GNN config (fitness fn)
│   ├── optimize.py             # OOA ⇄ GNN glue
│   ├── evaluate.py             # final model + baselines + plots
│   └── utils.py                # seeding, metrics, plotting
├── docs/                       # methodology.md · osprey_algorithm.md · architecture.md
├── notebooks/walkthrough.ipynb # interactive end-to-end demo
├── tests/                      # pytest suite (OOA, data, graph, models)
└── results/                    # generated metrics + figures + checkpoint
```

## Design choices at a glance

| Choice | Rationale |
|---|---|
| **From-scratch GNN layers** (no PyTorch-Geometric) | Transparent maths; no fragile compiled dependency; installs anywhere with just PyTorch. |
| **Disjoint-union graph** | Full-batch training over all rounds at once; message passing stays inside each round. |
| **Split by round** | Prevents neighbour leakage between train/val/test. |
| **Macro-F1 fitness** | Fair objective under heavy class imbalance (attacks are rare). |
| **Log-scaled lr / weight-decay in the search space** | Linear OOA moves explore multiplicative quantities uniformly. |
| **Synthetic generator + real-CSV loader** | Runs end-to-end with zero downloads, yet drops straight onto real WSN-DS. |
