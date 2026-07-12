# The Osprey Optimization Algorithm (OOA)

> Dehghani, M. & Trojovský, P. (2023). *Osprey optimization algorithm: A new
> bio-inspired metaheuristic algorithm for solving optimization problems.*
> Frontiers in Mechanical Engineering, 8, 1126450.

This document explains OOA in detail and how we map it onto GNN hyper-parameter
tuning. Implementation: [`src/osprey/optimizer.py`](../src/osprey/optimizer.py).

---

## 1. Biological inspiration

The **osprey** (*Pandion haliaetus*) is a fish-hawk. Its hunting strategy has two
distinct behaviours that OOA turns into the two pillars of any metaheuristic:

| Osprey behaviour | Optimisation role |
|---|---|
| Scans the sea, spots fish, and **dives to catch one** | **Exploration** — large jumps toward promising regions |
| **Carries the caught fish** to a safe spot to eat | **Exploitation** — small local refinement |

Each **osprey = one candidate solution** (a position vector in the search space).
Each **fish = another population member that is currently better**.

---

## 2. The algorithm

Let the population be `X = {x_1, …, x_N}` in a box `[lb, ub] ⊂ ℝ^d`, and let
`F(x)` be the fitness we **maximise**.

### Initialisation
Sample every osprey uniformly at random inside the box and evaluate its fitness.

### Phase 1 — Position identification & catching the fish (exploration)

For osprey `i`, define its set of "fish" as the members with **better fitness**:

```
FP_i = { x_k : F(x_k) > F(x_i) }
```

Select one fish `SF_i` at random from `FP_i` (if the set is empty, use the global
best). The osprey dives toward it:

$$ x_i^{new} = x_i + r \odot \big(SF_i - I \cdot x_i\big) $$

- `r` — a vector of `U(0,1)` random numbers (drawn per dimension),
- `I` — a random integer in `{1, 2}` (the paper's characteristic step factor),
- `⊙` — element-wise product.

**Greedy selection:** keep the new position only if it improves fitness.

### Phase 2 — Carrying the fish to a suitable position (exploitation)

The osprey then makes a small, **iteration-shrinking** local move:

$$ x_i^{new} = x_i + \frac{lb + r \odot (ub - lb)}{t} $$

where `t` is the current iteration (1-indexed), so the perturbation magnitude
decays like `1/t` — broad early exploration, fine late-stage exploitation. Again
accept greedily.

Repeat both phases for every osprey, for `t = 1 … T` iterations, tracking the
global best.

### Pseudocode

```
initialise X uniformly in [lb, ub];  evaluate F(X)
best ← argmax F(X)
for t = 1 … T:
    for i = 1 … N:
        # Phase 1 (exploration)
        SF ← random member with F > F(x_i)   (else global best)
        x_new ← x_i + r ⊙ (SF − I·x_i),  I ∈ {1,2}
        clip to [lb, ub];  if F(x_new) > F(x_i): x_i ← x_new
        # Phase 2 (exploitation)
        x_new ← x_i + (lb + r ⊙ (ub − lb)) / t
        clip to [lb, ub];  if F(x_new) > F(x_i): x_i ← x_new
    update global best
return best
```

**Complexity:** `O(N · T)` fitness evaluations (plus initialisation) — each of
which, here, trains a small GNN.

---

## 3. Why OOA for this problem?

- **Derivative-free** — GNN validation macro-F1 is non-differentiable w.r.t.
  discrete choices like layer type and depth; a metaheuristic handles them
  directly.
- **Few control parameters** — unlike PSO (inertia, cognitive/social weights) or
  GA (crossover/mutation rates), OOA has essentially only population size and
  iteration count, so there is little to mis-set.
- **Balanced search** — the two phases give a clean exploration/exploitation
  split, and the `1/t` decay anneals the search automatically.

---

## 4. Mapping to GNN hyper-parameters

OOA searches a continuous box; `config.decode_position` turns a position into
typed hyper-parameters (`config.SEARCH_SPACE`):

| Dimension | Range (continuous) | Decoded to |
|---|---|---|
| `hidden_dim` | 16 – 128 | nearest multiple of 8 (int) |
| `num_layers` | 2 – 4 | rounded int |
| `dropout` | 0.0 – 0.6 | float |
| `log_lr` | −4 – −2 | `lr = 10^{log_lr}` (1e-4 … 1e-2) |
| `log_wd` | −6 – −3 | `weight_decay = 10^{log_wd}` |
| `gnn_type` | 0 – 3 | index into `{gcn, sage, gat}` |

Log-scaling learning rate and weight decay lets the *linear* OOA moves explore
these *multiplicative* quantities uniformly across orders of magnitude.

**Fitness** = validation macro-F1 of a GNN trained (briefly) with the decoded
config. Maximising it drives the population toward architectures that generalise
best to unseen LEACH rounds.

---

## 5. Validation of the implementation

`tests/test_osprey.py` checks OOA on analytic benchmarks (Sphere, shifted
quadratic): it converges to the known optimum, its best-so-far curve is
monotonically non-decreasing, and it always respects the box constraints.
