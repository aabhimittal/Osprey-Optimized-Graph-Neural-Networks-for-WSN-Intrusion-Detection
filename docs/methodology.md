# Methodology — Step by Step

This document walks through the full pipeline, stage by stage, and explains the
*why* behind each design decision. Read it alongside the code — every section
names the module it corresponds to.

---

## 0. Problem statement

**Wireless Sensor Networks (WSNs)** are large collections of tiny, battery-powered
sensor nodes that self-organise into clusters and relay their readings to a
**Base Station (BS)**. The **LEACH** protocol rotates the role of *Cluster Head
(CH)* to spread the energy cost of long-range transmission. This openness makes
WSNs a soft target for **routing-layer attacks**:

| Attack | Behaviour | Effect |
|---|---|---|
| **Blackhole** | A node advertises itself as CH, attracts traffic, then **drops all** packets. | Data never reaches the BS. |
| **Grayhole** | Like a blackhole but drops **selectively / partially**. | Silent, intermittent data loss. |
| **Flooding** | Broadcasts a **huge number of control messages**. | Energy exhaustion, congestion. |
| **Scheduling (TDMA)** | Manipulates the **time-slot schedule**. | Collisions, unfair channel use. |

An **Intrusion Detection System (IDS)** must classify each node-round as *Normal*
or as one of these four attacks. This is a **node classification** problem on a
graph that changes every round.

**The two ideas this project combines:**

1. **Graph Neural Network (GNN)** — because an attack is only obvious *relative to
   its neighbours*, message passing along the communication topology is a natural
   inductive bias that plain tabular models lack.
2. **Osprey Optimization Algorithm (OOA)** — a GNN has many interacting
   hyper-parameters (depth, width, dropout, learning rate, layer type). Tuning
   them by hand is slow and fragile; a metaheuristic searches the space
   automatically. OOA is a recent (2023), simple, and effective bio-inspired
   optimiser.

---

## 1. Data (`src/data/`)

### 1.1 Schema — matching WSN-DS

We reproduce the schema of **WSN-DS** (Almomani et al., 2016), the standard
LEACH-based intrusion-detection benchmark. Each **row = one node in one LEACH
round**, described by 18 behavioural counters (`config.FEATURE_NAMES`): control
message counts (`ADV_*`, `JOIN_*`, `SCH_*`), data counts (`DATA_S`, `DATA_R`,
`Data_Sent_To_BS`), topology (`Dist_To_CH`, `Dist_CH_To_BS`, `Is_CH`), TDMA
(`Rank`), energy (`Consumed_Energy`), plus two engineered ratios
(`Rank_Ratio`, `Forward_Ratio`).

### 1.2 Synthetic generator (`synthetic.py`)

The real WSN-DS CSV must be downloaded separately, so the project ships a
**reproducible synthetic generator** that fabricates data with the *same schema
and the same behavioural signatures*. For each round it:

1. deploys `n_nodes` sensors uniformly in a square field;
2. elects ~5 % of them as Cluster Heads (LEACH's default probability *p*);
3. assigns each member to its nearest CH (forming the LEACH tree);
4. draws normal behavioural counters for every node; then
5. **overwrites** a fraction of nodes with attack-specific signatures, e.g. a
   blackhole gets `Forward_Ratio ≈ 0` while receiving lots of data.

A little **label noise** is injected for realism. The generator is unit-tested to
guarantee blackholes really do have a lower forward ratio than honest CHs.

> **Using the real dataset instead:** drop the official `WSN-DS.csv` into `data/`
> and the loader auto-detects it (via the `Attack type` column), maps its column
> names onto our schema (`loader.WSN_DS_COLUMN_MAP`), and the rest of the pipeline
> is unchanged.

### 1.3 Preprocessing (`loader.py`)

Loads the CSV, coerces to numeric, replaces NaN/inf, and **standard-scales** the
features (zero mean, unit variance) — essential for stable GNN training. Returns
a `Dataset` object carrying features, labels, round ids, CH membership and node
positions.

---

## 2. Graph construction (`src/graph/builder.py`)

Each round becomes a graph:

- **Nodes** = sensors active that round; **features** = the 18 counters.
- **Edges** = three complementary sources of relational structure:
  1. **Cluster edges** — every member ↔ its CH (the LEACH tree);
  2. **CH backbone** — CHs are interconnected (they all report to the BS and thus
     share context, letting the model compare CHs to one another);
  3. **Spatial k-NN** — each node ↔ its `k` nearest neighbours (radio proximity).
- **Self-loops** are added so every node keeps its own signal (GCN convention).

All per-round graphs are packed into **one disjoint-union graph** (block-diagonal
adjacency): node indices are offset per round, so message passing *never crosses
round boundaries*, yet the whole dataset trains full-batch with a single
`edge_index`. This is exactly PyTorch-Geometric's mini-batching trick,
implemented here in NumPy without the dependency.

**Split by round, not by node.** Whole rounds go to train / val / test. If we
split individual nodes, a node's own neighbours could leak across the split and
inflate the score. Splitting entire graphs keeps evaluation honest.

---

## 3. The GNN (`src/models/`)

### 3.1 Message passing from scratch (`layers.py`)

Rather than depend on PyTorch-Geometric, all layers are implemented directly so
the maths is visible. Every layer follows **message → aggregate → update** over
`edge_index`, aggregating with `index_add_` on the destination node (an explicit
sparse matrix-multiply):

- **GCN** — symmetric-normalised convolution
  `H' = σ(D̂^{-1/2} Â D̂^{-1/2} H W)`; each message is scaled by
  `1/√(deg_i·deg_j)`.
- **GraphSAGE** — mean aggregator with concatenation:
  `h_i' = σ(W·[h_i ‖ meanⱼ h_j])`.
- **GAT** — additive self-attention: learn per-edge weights
  `e_ij = LeakyReLU(aᵀ[Wh_i ‖ Wh_j])`, softmax-normalised over each node's
  incoming edges.

### 3.2 Classifier (`gnn.py`)

`GNNClassifier` stacks `num_layers` message-passing layers (LayerNorm + ReLU +
dropout between them) followed by a linear head that emits per-node logits over
the 5 classes. **The layer type, depth, width and dropout are all decided by the
Osprey search.**

---

## 4. Training a single configuration (`src/train.py`)

`train_gnn` performs **full-batch** node classification on the union graph:

- **Class-weighted cross-entropy** (inverse frequency) so rare attacks are not
  drowned out by the majority *Normal* class.
- **Adam** with the searched learning rate and weight decay.
- **Early stopping** on validation **macro-F1** (every class weighted equally —
  the fair metric for imbalanced IDS), restoring the best weights.

The same function is used both as the Osprey **fitness function** (short budget)
and to train the **final model** (long budget).

---

## 5. Osprey Optimization (`src/osprey/`, `src/optimize.py`)

See [`osprey_algorithm.md`](osprey_algorithm.md) for the full algorithm. In
brief, OOA maintains a population of *ospreys* (candidate hyper-parameter
vectors) and improves them over generations via two phases — **exploration**
(dive toward a better "fish") and **exploitation** (carry the fish a small,
shrinking step). We wrap it so:

```
osprey position ──decode──▶ GNN hyper-params ──train──▶ validation macro-F1 (fitness)
```

`config.decode_position` maps the continuous position vector to the *typed*
hyper-parameters (integers for depth/width, a categorical for layer type, log-scale
for lr/weight-decay). A small cache avoids retraining identical decoded configs.
OOA **maximises** validation macro-F1.

---

## 6. Final evaluation & baselines (`src/evaluate.py`)

The winning configuration is retrained to convergence and compared on the
held-out **test** set against:

- **Random Forest** on the raw node features (classic tabular IDS),
- **MLP** on the raw node features (a graph-blind neural net), and
- an **untuned GNN** with sensible default hyper-parameters.

Comparing *untuned GNN* vs *Osprey-tuned GNN* isolates the value the search adds;
comparing GNN vs the tabular models isolates the value of modelling topology. All
metrics (accuracy, macro-precision/recall/F1, weighted-F1), a confusion matrix, a
convergence curve and a model-comparison bar chart are written to `results/`.

---

## 7. Reproducibility

Every stage is seeded (`config.SEED = 42`). `python main.py --all` regenerates the
data, reruns the search and rewrites every artefact deterministically. Use
`--quick` for a ~1-2 minute smoke run with tiny budgets.
