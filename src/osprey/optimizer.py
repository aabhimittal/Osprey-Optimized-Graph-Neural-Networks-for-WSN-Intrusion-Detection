"""
Osprey Optimization Algorithm (OOA).

Reference
---------
Dehghani, M. & Trojovský, P. (2023). *Osprey optimization algorithm: A new
bio-inspired metaheuristic algorithm for solving optimization problems.*
Frontiers in Mechanical Engineering, 8, 1126450.

Biological inspiration
----------------------
The osprey (*Pandion haliaetus*, a fish-hawk) hunts in two behavioural phases,
which OOA turns into an exploration phase and an exploitation phase:

**Phase 1 — position identification & catching the fish (exploration).**
The osprey scans the sea surface, detects fish, selects one and dives to catch
it.  For osprey ``i`` the set of "fish" is the population members with *better*
fitness.  A fish ``SF_i`` is picked at random from that set (or the global best
if none is better) and the osprey moves toward it:

    x_new = x_i + r * (SF_i - I * x_i)

with ``r ~ U(0,1)`` drawn per-dimension and ``I`` a random integer in {1, 2}.
The move is accepted only if it improves fitness (greedy selection).

**Phase 2 — carrying the fish to a suitable position (exploitation).**
Having caught the fish the osprey carries it to a safe feeding spot — a small,
iteration-shrinking local step that refines the solution:

    x_new = x_i + (lb + r*(ub - lb)) / t          (paper form)

where ``t`` is the current iteration (1-indexed), so steps shrink as ``1/t``.
Again accepted greedily.

This module implements OOA as a **maximiser** operating on a bounded box; the
fitness function is supplied by the caller (here: validation macro-F1 of a GNN).
It is written generically and is unit-tested on analytic functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class OspreyResult:
    best_position: np.ndarray
    best_fitness: float
    history: List[float] = field(default_factory=list)      # best-so-far per iteration
    mean_history: List[float] = field(default_factory=list)  # population mean per iteration
    evaluations: int = 0


class OspreyOptimizer:
    """A faithful, maximising implementation of the Osprey Optimization Algorithm.

    Parameters
    ----------
    fitness_fn : Callable[[np.ndarray], float]
        Objective to **maximise**.  Receives a position vector, returns a scalar.
    lower_bounds, upper_bounds : np.ndarray
        Box constraints, shape ``(dim,)``.
    n_ospreys : int
        Population size.
    n_iterations : int
        Number of generations.
    seed : int
        RNG seed for reproducibility.
    verbose : bool
        Print per-iteration progress.
    """

    def __init__(self, fitness_fn: Callable[[np.ndarray], float],
                 lower_bounds: np.ndarray, upper_bounds: np.ndarray,
                 n_ospreys: int = 8, n_iterations: int = 12,
                 seed: int = 42, verbose: bool = True):
        self.fitness_fn = fitness_fn
        self.lb = np.asarray(lower_bounds, dtype=float)
        self.ub = np.asarray(upper_bounds, dtype=float)
        assert self.lb.shape == self.ub.shape, "bound shapes must match"
        self.dim = self.lb.shape[0]
        self.n = n_ospreys
        self.T = n_iterations
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self._evals = 0

    # --------------------------------------------------------------------- #
    def _clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lb, self.ub)

    def _evaluate(self, x: np.ndarray) -> float:
        self._evals += 1
        return float(self.fitness_fn(self._clip(x)))

    # --------------------------------------------------------------------- #
    def optimize(self) -> OspreyResult:
        # --- initialise population uniformly in the box --------------------
        X = self.rng.uniform(self.lb, self.ub, size=(self.n, self.dim))
        fit = np.array([self._evaluate(X[i]) for i in range(self.n)])

        best_idx = int(np.argmax(fit))
        best_x = X[best_idx].copy()
        best_f = float(fit[best_idx])

        history, mean_history = [], []

        for t in range(1, self.T + 1):
            for i in range(self.n):
                # ---------- Phase 1: exploration (catch a fish) ----------
                # "fish" = population members with strictly better fitness
                better = np.where(fit > fit[i])[0]
                if better.size > 0:
                    sf = X[self.rng.choice(better)]           # selected fish
                else:
                    sf = best_x                               # none better -> global best
                r = self.rng.uniform(0, 1, self.dim)
                I = self.rng.integers(1, 3)                   # 1 or 2
                x_new = X[i] + r * (sf - I * X[i])
                x_new = self._clip(x_new)
                f_new = self._evaluate(x_new)
                if f_new > fit[i]:                            # greedy accept
                    X[i], fit[i] = x_new, f_new

                # ---------- Phase 2: exploitation (carry the fish) ----------
                r2 = self.rng.uniform(0, 1, self.dim)
                x_new = X[i] + (self.lb + r2 * (self.ub - self.lb)) / t
                x_new = self._clip(x_new)
                f_new = self._evaluate(x_new)
                if f_new > fit[i]:
                    X[i], fit[i] = x_new, f_new

            # track global best
            it_best = int(np.argmax(fit))
            if fit[it_best] > best_f:
                best_f = float(fit[it_best])
                best_x = X[it_best].copy()

            history.append(best_f)
            mean_history.append(float(fit.mean()))
            if self.verbose:
                print(f"  [OOA] iter {t:2d}/{self.T} | best F1={best_f:.4f} "
                      f"| pop-mean={fit.mean():.4f} | evals={self._evals}")

        return OspreyResult(
            best_position=best_x,
            best_fitness=best_f,
            history=history,
            mean_history=mean_history,
            evaluations=self._evals,
        )
