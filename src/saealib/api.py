"""High-level API: minimize function and Result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

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
from saealib.problem import non_dominated_sort
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ib import IndividualBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.manager import LocalSurrogateManager, SurrogateManager
from saealib.surrogate.rbf import RBFsurrogate, gaussian_kernel
from saealib.termination import Termination
from saealib.termination import max_fe as max_fe_cond

if TYPE_CHECKING:
    from saealib.algorithms.base import Algorithm
    from saealib.problem import Problem
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.base import Surrogate


@dataclass
class Result:
    """Optimization result returned by :func:`minimize`.

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


def _resolve_algorithm(algorithm: Union[str, "Algorithm"]) -> "Algorithm":
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
            f"Unknown algorithm: {algorithm!r}. Use 'GA', 'PSO', or an Algorithm instance."
        )
    return algorithm


def _resolve_surrogate(
    surrogate: Union[str, "Surrogate", SurrogateManager, None],
    problem: "Problem",
    n_neighbors: int,
) -> SurrogateManager:
    if surrogate is None:
        raise ValueError(
            "surrogate=None is not supported. Use 'rbf' or a Surrogate/SurrogateManager instance."
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
            f"Unknown surrogate: {surrogate!r}. Use 'rbf' or a Surrogate/SurrogateManager instance."
        )
    # Surrogate instance
    return LocalSurrogateManager(
        surrogate,
        MeanPrediction(weights=problem.weight),
        n_neighbors=n_neighbors,
    )


def _resolve_strategy(
    strategy: Union[str, "OptimizationStrategy", None],
    pop_size: int,
) -> "OptimizationStrategy":
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
            f"Unknown strategy: {strategy!r}. Use 'ib', 'gb', 'ps', or an OptimizationStrategy instance."
        )
    return strategy


def _build_result(ctx: OptimizationContext) -> Result:
    archive_x = ctx.archive.get_array("x")
    archive_f = ctx.archive.get_array("f")
    weight = ctx.problem.weight

    if ctx.problem.n_obj == 1:
        scores = archive_f @ weight
        best_idx = int(np.argmax(scores))
        X = archive_x[best_idx]
        F = archive_f[best_idx]
    else:
        fronts = non_dominated_sort(archive_f, weight)
        pareto_idx = fronts[0]
        X = archive_x[pareto_idx]
        F = archive_f[pareto_idx]

    return Result(X=X, F=F, fe=ctx.fe, gen=ctx.gen, ctx=ctx)


def minimize(
    problem: "Problem",
    algorithm: Union[str, "Algorithm"] = "GA",
    *,
    surrogate: Union[str, "Surrogate", SurrogateManager, None] = "rbf",
    strategy: Union[str, "OptimizationStrategy", None] = "ib",
    max_fe: Union[int, None] = None,
    pop_size: Union[int, None] = None,
    n_neighbors: int = 50,
    seed: Union[int, None] = None,
    verbose: bool = True,
) -> Result:
    """Run surrogate-assisted optimization with sensible defaults.

    Parameters
    ----------
    problem : Problem
        Optimization problem.
    algorithm : str or Algorithm
        ``'GA'``, ``'PSO'``, or an :class:`Algorithm` instance. Default: ``'GA'``.
    surrogate : str, Surrogate, SurrogateManager, or None
        ``'rbf'``, a :class:`Surrogate` instance, or a :class:`SurrogateManager`
        instance. Default: ``'rbf'``.
    strategy : str or OptimizationStrategy
        ``'ib'`` (IndividualBased), ``'gb'`` (GenerationBased), ``'ps'``
        (PreSelection), or an :class:`OptimizationStrategy` instance. Default:
        ``'ib'``.
    max_fe : int or None
        Maximum true function evaluations. Default: ``200 * problem.dim``.
    pop_size : int or None
        Population size. Default: ``4 * problem.dim``.
    n_neighbors : int
        Nearest neighbours used by :class:`LocalSurrogateManager`. Default: 50.
    seed : int or None
        Random seed passed to :class:`LHSInitializer`.
    verbose : bool
        If ``False``, suppress per-generation log output. Default: ``True``.

    Returns
    -------
    Result
        Optimization result with best solution, objective value, and context.

    Examples
    --------
    >>> from saealib import minimize, Problem
    >>> import numpy as np
    >>> problem = Problem(func=lambda x: np.sum(x**2), dim=5, n_obj=1,
    ...                   weight=np.array([-1.0]), lb=[-5.0]*5, ub=[5.0]*5)
    >>> result = minimize(problem, "GA", max_fe=500, seed=0)
    >>> result.X, result.F
    """
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
