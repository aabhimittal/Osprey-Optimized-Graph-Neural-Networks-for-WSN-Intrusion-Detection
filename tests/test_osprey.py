"""Tests for the Osprey Optimization Algorithm on analytic functions.

If OOA works, it should locate the optimum of standard benchmark functions.
We optimise (as a maximiser) the negatives of the Sphere and Rastrigin
functions, whose maxima are at the origin with value 0.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.osprey.optimizer import OspreyOptimizer


def test_osprey_minimises_sphere():
    # maximise -sum(x^2)  -> optimum at 0 with value 0
    dim = 5
    lb, ub = -5.12 * np.ones(dim), 5.12 * np.ones(dim)

    opt = OspreyOptimizer(
        fitness_fn=lambda x: -float(np.sum(x ** 2)),
        lower_bounds=lb, upper_bounds=ub,
        n_ospreys=20, n_iterations=60, seed=0, verbose=False,
    )
    res = opt.optimize()

    assert res.best_fitness > -0.5, f"did not converge: {res.best_fitness}"
    assert np.linalg.norm(res.best_position) < 1.0


def test_osprey_convergence_is_monotone_nondecreasing():
    dim = 3
    lb, ub = -10 * np.ones(dim), 10 * np.ones(dim)
    opt = OspreyOptimizer(
        fitness_fn=lambda x: -float(np.sum((x - 1.0) ** 2)),
        lower_bounds=lb, upper_bounds=ub,
        n_ospreys=15, n_iterations=40, seed=1, verbose=False,
    )
    res = opt.optimize()
    # best-so-far history must never decrease
    h = np.array(res.history)
    assert np.all(np.diff(h) >= -1e-9)
    # should get close to x = [1,1,1]
    assert np.linalg.norm(res.best_position - 1.0) < 1.5


def test_osprey_respects_bounds():
    dim = 4
    lb, ub = np.zeros(dim), np.ones(dim)
    opt = OspreyOptimizer(
        fitness_fn=lambda x: float(np.sum(x)),  # pushes toward the upper bound
        lower_bounds=lb, upper_bounds=ub,
        n_ospreys=10, n_iterations=20, seed=2, verbose=False,
    )
    res = opt.optimize()
    assert np.all(res.best_position <= ub + 1e-9)
    assert np.all(res.best_position >= lb - 1e-9)
