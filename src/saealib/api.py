"""High-level API: minimize / maximize functions and Result dataclass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from saealib.acquisition.mean import MeanPrediction
from saealib.algorithms.ga import GA
from saealib.algorithms.pso import PSO
from saealib.callback import GenerationStartEvent, logging_generation
from saealib.context import OptimizationContext
from saealib.execution.initializer import LHSInitializer
from saealib.operators.crossover import CrossoverBLXAlpha
from saealib.operators.mutation import MutationUniform
from saealib.operators.selection import SequentialSelection, TruncationSelection
from saealib.optimizer import Optimizer
from saealib.comparators import non_dominated_sort
from saealib.problem import Problem
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.manager import LocalSurrogateManager, SurrogateManager
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel
from saealib.termination import Termination
from saealib.termination import max_fe as max_fe_cond

if TYPE_CHECKING:
    from saealib.algorithms.base import Algorithm
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.base import Surrogate


@dataclass
class Result:
    """Optimization result returned by :func:`minimize` / :func:`maximize`.

    Attributes
    ----------
    X : np.ndarray
        Best design variables. Shape ``(dim,)`` for single-objective,
        ``(n_pareto, dim)`` for multi-objective.
    F : np.ndarray
        Best objective values. Shape ``(n_obj,)`` for single-objective,
        ``(n_pareto, n_obj)`` for multi-objective.
    fe : int
        Total number of true function evaluations used.
    gen : int
        Total number of generations completed.
    ctx : OptimizationContext
        Full optimization context providing access to the archive and more.
    """

    X: np.ndarray
    F: np.ndarray
    fe: int
    gen: int
    ctx: OptimizationContext


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_problem(
    func: Callable | Problem,
    dim: int | None,
    lb,
    ub,
    n_obj: int,
    weight: np.ndarray,
) -> Problem:
    """Return a Problem, building one from a callable if needed."""
    if isinstance(func, Problem):
        return func
    if dim is None or lb is None or ub is None:
        raise ValueError("dim, lb, and ub are required when func is a callable.")
    return Problem(func=func, dim=dim, n_obj=n_obj, weight=weight, lb=lb, ub=ub)


def _resolve_algorithm(algorithm: str | Algorithm) -> Algorithm:
    if isinstance(algorithm, str):
        name = algorithm.upper()
        if name == "GA":
            return GA(
                crossover=CrossoverBLXAlpha(crossover_rate=0.7, alpha=0.4),
                mutation=MutationUniform(mutation_rate=0.3),
                parent_selection=SequentialSelection(),
                survivor_selection=TruncationSelection(),
            )
        if name == "PSO":
            return PSO()
        raise ValueError(
            f"Unknown algorithm: {algorithm!r}. "
            "Use 'GA', 'PSO', or an Algorithm instance."
        )
    return algorithm


def _resolve_surrogate(
    surrogate: str | Surrogate | SurrogateManager | None,
    problem: Problem,
    n_neighbors: int,
) -> SurrogateManager:
    if surrogate is None:
        raise ValueError(
            "surrogate=None is not supported. "
            "Use 'rbf' or a Surrogate/SurrogateManager instance."
        )
    if isinstance(surrogate, SurrogateManager):
        return surrogate
    if isinstance(surrogate, str):
        if surrogate.lower() == "rbf":
            rbf = RBFsurrogate(gaussian_kernel, problem.dim)
            return LocalSurrogateManager(
                rbf,
                MeanPrediction(weights=problem.weight),
                n_neighbors=n_neighbors,
            )
        raise ValueError(
            f"Unknown surrogate: {surrogate!r}. "
            "Use 'rbf' or a Surrogate/SurrogateManager instance."
        )
    return LocalSurrogateManager(
        surrogate,
        MeanPrediction(weights=problem.weight),
        n_neighbors=n_neighbors,
    )


def _resolve_strategy(
    strategy: str | OptimizationStrategy | None,
    pop_size: int,
) -> OptimizationStrategy:
    if strategy is None or (isinstance(strategy, str) and strategy.lower() == "ib"):
        return IndividualBasedStrategy(evaluation_ratio=0.1)
    if isinstance(strategy, str):
        name = strategy.lower()
        if name == "gb":
            return GenerationBasedStrategy(gen_ctrl=5)
        if name == "ps":
            n_select = max(1, pop_size // 10)
            return PreSelectionStrategy(n_candidates=pop_size, n_select=n_select)
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            "Use 'ib', 'gb', 'ps', or an OptimizationStrategy instance."
        )
    return strategy


def _build_result(ctx: OptimizationContext) -> Result:
    archive_x = ctx.archive.get_array("x")
    archive_f = ctx.archive.get_array("f")
    weight = ctx.problem.weight

    if ctx.problem.n_obj == 1:
        scores = archive_f @ weight
        best_idx = int(np.argmax(scores))
        best_x = archive_x[best_idx]
        best_f = archive_f[best_idx]
    else:
        fronts = non_dominated_sort(archive_f, weight)
        pareto_idx = fronts[0]
        best_x = archive_x[pareto_idx]
        best_f = archive_f[pareto_idx]

    return Result(X=best_x, F=best_f, fe=ctx.fe, gen=ctx.gen, ctx=ctx)


def _run(
    problem: Problem,
    algorithm: str | Algorithm,
    surrogate: str | Surrogate | SurrogateManager | None,
    strategy: str | OptimizationStrategy | None,
    max_fe: int | None,
    pop_size: int | None,
    n_neighbors: int,
    seed: int | None,
    verbose: bool,
) -> Result:
    dim = problem.dim
    if pop_size is None:
        pop_size = 4 * dim
    if max_fe is None:
        max_fe = 200 * dim

    alg = _resolve_algorithm(algorithm)
    sm = _resolve_surrogate(surrogate, problem, n_neighbors)
    strat = _resolve_strategy(strategy, pop_size)

    initializer = LHSInitializer(
        n_init_archive=5 * dim,
        n_init_population=pop_size,
        seed=seed,
    )
    termination = Termination(max_fe_cond(max_fe))

    opt = (
        Optimizer(problem)
        .set_initializer(initializer)
        .set_algorithm(alg)
        .set_termination(termination)
        .set_surrogate_manager(sm)
        .set_strategy(strat)
    )

    if not verbose:
        opt.cbmanager.unregister(GenerationStartEvent, logging_generation)

    ctx = opt.run()
    return _build_result(ctx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def minimize(
    func: Callable | Problem,
    algorithm: str | Algorithm = "GA",
    *,
    dim: int | None = None,
    lb=None,
    ub=None,
    n_obj: int = 1,
    surrogate: str | Surrogate | SurrogateManager | None = "rbf",
    strategy: str | OptimizationStrategy | None = "ib",
    max_fe: int | None = None,
    pop_size: int | None = None,
    n_neighbors: int = 50,
    seed: int | None = None,
    verbose: bool = True,
) -> Result:
    """Run surrogate-assisted minimization.

    Parameters
    ----------
    func : callable or Problem
        Objective function ``f(x) -> float | array``, or a fully configured
        :class:`Problem` instance (in which case ``dim``, ``lb``, ``ub``, and
        ``n_obj`` are ignored).
    algorithm : str or Algorithm
        ``'GA'``, ``'PSO'``, or an :class:`Algorithm` instance. Default: ``'GA'``.
    dim : int, optional
        Number of design variables. Required when *func* is a callable.
    lb : array-like, optional
        Lower bounds of length *dim*. Required when *func* is a callable.
    ub : array-like, optional
        Upper bounds of length *dim*. Required when *func* is a callable.
    n_obj : int
        Number of objectives. Ignored when *func* is a :class:`Problem`. Default: 1.
    surrogate : str, Surrogate, SurrogateManager, or None
        ``'rbf'``, a :class:`Surrogate`, or a :class:`SurrogateManager`.
        Default: ``'rbf'``.
    strategy : str or OptimizationStrategy
        ``'ib'``, ``'gb'``, ``'ps'``, or an :class:`OptimizationStrategy`.
        Default: ``'ib'``.
    max_fe : int or None
        Maximum true function evaluations. Default: ``200 * dim``.
    pop_size : int or None
        Population size. Default: ``4 * dim``.
    n_neighbors : int
        Nearest neighbours for :class:`LocalSurrogateManager`. Default: 50.
    seed : int or None
        Random seed for :class:`LHSInitializer`.
    verbose : bool
        If ``False``, suppress per-generation log output. Default: ``True``.

    Returns
    -------
    Result

    Examples
    --------
    >>> from saealib import minimize
    >>> import numpy as np
    >>> result = minimize(lambda x: np.sum(x**2), dim=5, lb=[-5]*5, ub=[5]*5,
    ...                   max_fe=500, seed=0, verbose=False)
    >>> result.X, result.F
    """
    weight = np.full(n_obj, -1.0)
    problem = _ensure_problem(func, dim, lb, ub, n_obj, weight)
    return _run(
        problem,
        algorithm,
        surrogate,
        strategy,
        max_fe,
        pop_size,
        n_neighbors,
        seed,
        verbose,
    )


def maximize(
    func: Callable | Problem,
    algorithm: str | Algorithm = "GA",
    *,
    dim: int | None = None,
    lb=None,
    ub=None,
    n_obj: int = 1,
    surrogate: str | Surrogate | SurrogateManager | None = "rbf",
    strategy: str | OptimizationStrategy | None = "ib",
    max_fe: int | None = None,
    pop_size: int | None = None,
    n_neighbors: int = 50,
    seed: int | None = None,
    verbose: bool = True,
) -> Result:
    """Run surrogate-assisted maximization.

    Identical to :func:`minimize` except that all objectives are maximized
    (``weight = +1``).

    Parameters
    ----------
    func : callable or Problem
        Objective function ``f(x) -> float | array``, or a fully configured
        :class:`Problem` instance.
    algorithm : str or Algorithm
        ``'GA'``, ``'PSO'``, or an :class:`Algorithm` instance. Default: ``'GA'``.
    dim : int, optional
        Number of design variables. Required when *func* is a callable.
    lb : array-like, optional
        Lower bounds of length *dim*. Required when *func* is a callable.
    ub : array-like, optional
        Upper bounds of length *dim*. Required when *func* is a callable.
    n_obj : int
        Number of objectives. Ignored when *func* is a :class:`Problem`. Default: 1.
    surrogate : str, Surrogate, SurrogateManager, or None
        ``'rbf'``, a :class:`Surrogate`, or a :class:`SurrogateManager`.
        Default: ``'rbf'``.
    strategy : str or OptimizationStrategy
        ``'ib'``, ``'gb'``, ``'ps'``, or an :class:`OptimizationStrategy`.
        Default: ``'ib'``.
    max_fe : int or None
        Maximum true function evaluations. Default: ``200 * dim``.
    pop_size : int or None
        Population size. Default: ``4 * dim``.
    n_neighbors : int
        Nearest neighbours for :class:`LocalSurrogateManager`. Default: 50.
    seed : int or None
        Random seed for :class:`LHSInitializer`.
    verbose : bool
        If ``False``, suppress per-generation log output. Default: ``True``.

    Returns
    -------
    Result

    Examples
    --------
    >>> from saealib import maximize
    >>> import numpy as np
    >>> result = maximize(lambda x: -np.sum(x**2) + 10, dim=5, lb=[-5]*5, ub=[5]*5,
    ...                   max_fe=500, seed=0, verbose=False)
    >>> result.X, result.F
    """
    weight = np.full(n_obj, +1.0)
    problem = _ensure_problem(func, dim, lb, ub, n_obj, weight)
    return _run(
        problem,
        algorithm,
        surrogate,
        strategy,
        max_fe,
        pop_size,
        n_neighbors,
        seed,
        verbose,
    )
