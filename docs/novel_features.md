# Novel Features — Beyond the Core Pipeline

The core pipeline (data → graph → Osprey-tuned GNN → evaluation) is documented
in [`methodology.md`](methodology.md). This document covers the four research
extensions layered on top, each addressing a question a *deployable* WSN IDS
must answer.

---

## 1. Resource-aware multi-objective Osprey search

**Question: is the most accurate model actually deployable on a sensor node?**

WSN hardware is battery- and CPU-constrained, so raw accuracy is the wrong
objective — what matters is accuracy *per joule*. The Osprey fitness can
therefore be switched from pure macro-F1 to a scalarised two-objective form:

```
fitness = macro_F1 − efficiency_weight · (n_parameters / PARAM_BUDGET)
```

- `n_parameters` is the trainable-parameter count of the model the osprey's
  position decodes to (`src/optimize.py::count_parameters`);
- `PARAM_BUDGET` (config.py) normalises the penalty to roughly [0, 1] — it is
  the parameter count of the largest model the search space can express;
- `efficiency_weight` (default 0.15 via `--multi-objective`) sets the
  exchange rate: 0.15 means "1 % of F1 is worth ~1 % of the parameter budget".

This turns OOA into a lightweight **hardware-aware neural-architecture
search**: the population is pushed toward the accuracy/size Pareto front, and
the winning model is typically several times smaller for near-identical F1.

```bash
python main.py --optimize --evaluate --multi-objective
```

The chosen model's parameter count is now recorded in
`results/best_config.json` for either mode.

---

## 2. Explainability: *what* and *who* (`src/explain.py`)

**Question: why was this node flagged?** An IDS alarm nobody can interpret is
an alarm nobody trusts. Two complementary explanations are produced:

### 2a. Feature saliency — *what the model looked at*

For every test node we take the gradient of its predicted-class logit w.r.t.
the input features and average |gradients| per predicted class, giving a
`classes × features` importance map (`results/explainability.png`).

This doubles as a **model audit**: the map should recover the true attack
physics — `Forward_Ratio`/`DATA_R` for Blackhole/Grayhole, `ADV_S`/`Send_Code`
for Flooding, `Rank`/`Rank_Ratio`/`SCH_*` for Scheduling. If it highlights
something unrelated, the model has learned a shortcut and should not be
trusted, however good its F1.

### 2b. Neighbour occlusion — *who provided the evidence*

Saliency explains features; graphs also have **relational evidence**. For a
sample of flagged nodes we delete one neighbour's edges at a time, re-run the
model, and record the drop in the predicted-class logit. Neighbours whose
removal collapses the prediction are the ones whose traffic exposed the
attacker — e.g. the cluster members whose unforwarded packets betray a
blackhole CH. Ranked attributions land in `results/explanations.json`.

```bash
python main.py --explain
```

---

## 3. Cross-round trust / reputation scoring (`src/trust.py`)

**Question: which physical nodes are compromised — not just which rows are
anomalous?**

A single LEACH round yields one noisy verdict; real compromised nodes persist.
The generator now plants **persistent attackers**
(`DataConfig.persistent_attacker_fraction`, default 60 % of the malicious
budget) that keep the same attack type every round, alongside transient
per-round attackers.

Per-round GNN posteriors are folded into a per-node **trust index** with an
exponential moving average:

```
trust_t(v) = α · trust_{t−1}(v) + (1 − α) · P(Normal | v, round t),   trust_0 = 1
```

Honest nodes hover near 1; persistent attackers are driven monotonically
toward 0 as evidence accumulates. Nodes crossing the quarantine threshold
(default 0.5) would be excluded from CH election — the natural WSN response.

`results/trust_report.json` quantifies the gain over one-shot detection
(precision/recall/F1 on the planted persistent attackers, plus **mean
detection latency** in rounds), and `results/trust_scores.png` shows the
trajectories separating.

```bash
python main.py --trust
```

---

## 4. Adversarial / noise robustness (`src/robustness.py`)

**Question: does detection survive a non-cooperative world?**

Two perturbation sweeps on the standard-scaled test features, ε ∈ [0, 1] std
units, evaluated for the tuned GNN **and** the Random-Forest baseline:

- **Gaussian noise on all test nodes** — models sensor/radio noise;
- **Evasion: only malicious nodes perturbed** — attackers deliberately blur
  their own counters while the rest of the network stays clean.

The structural hypothesis under evasion: a tabular model sees only the row the
attacker controls, but the GNN also reads the *neighbours'* rows — the missing
traffic at honest members still betrays a blackhole even if the blackhole
edits its own counters. The two curves in `results/robustness.png` make this
comparison explicit.

```bash
python main.py --robustness
```

---

## Putting it together

`python main.py --all` now runs all seven stages:
generate → optimize → evaluate → **explain → trust → robustness**, writing:

| Artefact | Feature |
|---|---|
| `results/best_config.json` (+ `n_parameters`) | multi-objective search |
| `results/explainability.png`, `explanations.json` | explainability |
| `results/trust_scores.png`, `trust_report.json` | trust scoring |
| `results/robustness.png`, `robustness.json` | robustness |

All four features are unit-tested in `tests/test_novel_features.py`.
