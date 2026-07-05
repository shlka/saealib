"""High-level API: minimize / maximize functions and Result dataclass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from saealib.acquisition.mean import MeanPrediction
from saealib.callback import GenerationStartEvent, logging_generation
from saealib.context import OptimizationState
from saealib.exceptions import ValidationError
from saealib.execution.initializer import LHSInitializer
from saealib.optimizer import Optimizer
from saealib.problem import Problem
from saealib.strategies.gb import GenerationBasedStrategy
from saealib.strategies.ps import PreSelectionStrategy
from saealib.surrogate.manager import LocalSurrogateManager, SurrogateManager
from saealib.termination import Termination
from saealib.termination import max_fe as max_fe_cond

if TYPE_CHECKING:
    from pathlib import Path

    from saealib.algorithms.base import Algorithm
    from saealib.strategies.base import OptimizationStrategy
    from saealib.surrogate.base import Surrogate


# Sentinel distinguishing "argument omitted" (defer to Optimizer's own
# default resolution) from an explicit ``None`` passed by the caller (which
# keeps its pre-existing meaning, e.g. surrogate=None raises ValidationError).
class _UnsetType:
    def __repr__(self) -> str:
        return "UNSET"


_UNSET = _UnsetType()


@dataclass
class Result:
    """Optimization result returned by :func:`minimize` / :func:`maximize`.

    Attributes
    ----------
    x : np.ndarray
        Best design variables. Shape ``(dim,)`` for single-objective,
        ``(n_pareto, dim)`` for multi-objective.
    f : np.ndarray
        Best objective values. Shape ``(n_obj,)`` for single-objective,
        ``(n_pareto, n_obj)`` for multi-objective.
    fe : int
        Total number of true function evaluations used.
    gen : int
        Total number of generations completed.
    ctx : OptimizationState
        Full optimization context providing access to the archive and more.
    """

    x: np.ndarray
    f: np.ndarray
    fe: int
    gen: int
    ctx: OptimizationState


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_direction(
    direction: np.ndarray | list[str] | None,
    n_obj: int,
    default: float,
) -> np.ndarray:
    """Convert direction argument to a ±1 float array."""
    if direction is None:
        return np.full(n_obj, default)
    if isinstance(direction, np.ndarray):
        return direction
    _map = {"minimize": -1.0, "maximize": 1.0}
    try:
        return np.array([_map[d] for d in direction])
    except KeyError as e:
        raise ValidationError(
            f"Unknown direction {e}. Use 'minimize' or 'maximize'."
        ) from e


def _ensure_problem(
    func: Callable | Problem,
    dim: int | None,
    lb,
    ub,
    n_obj: int,
    direction: np.ndarray,
) -> Problem:
    """Return a Problem, building one from a callable if needed."""
    if isinstance(func, Problem):
        return func
    if dim is None or lb is None or ub is None:
        raise ValidationError("dim, lb, and ub are required when func is a callable.")
    return Problem(func=func, dim=dim, n_obj=n_obj, direction=direction, lb=lb, ub=ub)


def _resolve_algorithm(algorithm: str | Algorithm | None) -> Algorithm | None:
    if isinstance(algorithm, str):
        from saealib.defaults import load_defaults
        from saealib.registry import build, get

        name = algorithm.upper()
        if name == "GA":
            spec = load_defaults()["presets"]["ga_rbf_ib"]["algorithm"]
            return build(spec)
        if name == "PSO":
            return get("PSO")()
        raise ValidationError(
            f"Unknown algorithm: {algorithm!r}. "
            "Use 'GA', 'PSO', or an Algorithm instance."
        )
    return algorithm


def _resolve_surrogate(
    surrogate: str | Surrogate | SurrogateManager | None,
    problem: Problem,
) -> SurrogateManager:
    if surrogate is None:
        raise ValidationError(
            "surrogate=None is not supported. "
            "Use 'rbf' or a Surrogate/SurrogateManager instance."
        )
    if isinstance(surrogate, SurrogateManager):
        return surrogate
    if isinstance(surrogate, str):
        if surrogate.lower() == "rbf":
            from saealib.defaults import load_defaults

            spec = load_defaults()["presets"]["ga_rbf_ib"]["surrogate_manager"]
            return Optimizer._build_surrogate_manager_from_spec(
                spec, problem.dim, problem.direction
            )
        raise ValidationError(
            f"Unknown surrogate: {surrogate!r}. "
            "Use 'rbf' or a Surrogate/SurrogateManager instance."
        )
    from saealib.surrogate.training_set import KNNObjectiveSet

    return LocalSurrogateManager(
        surrogate,
        MeanPrediction(direction=problem.direction),
        training_set=KNNObjectiveSet(),
    )


def _resolve_strategy(
    strategy: str | OptimizationStrategy | None,
    pop_size: int,
) -> OptimizationStrategy | None:
    if isinstance(strategy, str):
        name = strategy.lower()
        if name == "ib":
            from saealib.defaults import load_defaults
            from saealib.registry import build

            spec = load_defaults()["presets"]["ga_rbf_ib"]["strategy"]
            return build(spec)
        if name == "gb":
            return GenerationBasedStrategy(gen_ctrl=5)
        if name == "ps":
            n_select = max(1, pop_size // 10)
            return PreSelectionStrategy(n_candidates=pop_size, n_select=n_select)
        raise ValidationError(
            f"Unknown strategy: {strategy!r}. "
            "Use 'ib', 'gb', 'ps', or an OptimizationStrategy instance."
        )
    return strategy


def _build_result(ctx: OptimizationState) -> Result:
    archive_x = ctx.archive.get_array("x")
    archive_f = ctx.archive.get_array("f")
    archive_cv = ctx.archive.get_array("cv")
    direction = ctx.problem.direction
    eps = ctx.problem.eps_cv

    feasible = np.where(archive_cv <= eps)[0]
    pool = feasible if len(feasible) else np.array([int(np.argmin(archive_cv))])

    if ctx.problem.n_obj == 1:
        scores = archive_f[pool] @ direction
        best_idx = pool[int(np.argmax(scores))]
        best_x = archive_x[best_idx]
        best_f = archive_f[best_idx]
    else:
        if len(ctx.pareto_archive) > 0:
            best_x = ctx.pareto_archive.get_array("x")
            best_f = ctx.pareto_archive.get_array("f")
        else:
            # Fallback: pareto_archive is empty (should not happen in normal use)
            from saealib.comparators import non_dominated_sort

            _, fronts = non_dominated_sort(archive_f[pool], direction=direction)
            pareto_idx = pool[fronts[0]]
            best_x = archive_x[pareto_idx]
            best_f = archive_f[pareto_idx]

    return Result(x=best_x, f=best_f, fe=ctx.fe, gen=ctx.gen, ctx=ctx)


def _run(
    problem: Problem,
    algorithm: str | Algorithm | None,
    surrogate: str | Surrogate | SurrogateManager | None,
    strategy: str | OptimizationStrategy | None,
    max_fe: int | None,
    pop_size: int | None,
    seed: int | None,
    verbose: bool,
    preset: str | Path | dict | None,
) -> Result:
    dim = problem.dim
    if pop_size is None:
        pop_size = 4 * dim
    if max_fe is None:
        max_fe = 200 * dim

    initializer = LHSInitializer(
        n_init_archive=5 * dim,
        n_init_population=pop_size,
        seed=seed,
    )
    termination = Termination(max_fe_cond(max_fe))

    opt = Optimizer(problem).set_initializer(initializer).set_termination(termination)

    if preset is not None:
        opt.set_preset(preset)

    # Arguments left at _UNSET are not passed to set_*(); Optimizer.run()'s
    # _resolve_defaults() then fills them from the library's bundled preset.
    if algorithm is not _UNSET:
        # An explicit None resolves to None here, same as never calling
        # set_algorithm(); Optimizer._resolve_defaults() then fills it in.
        opt.set_algorithm(_resolve_algorithm(algorithm))  # type: ignore
    if surrogate is not _UNSET:
        opt.set_surrogate_manager(_resolve_surrogate(surrogate, problem))
    if strategy is not _UNSET:
        opt.set_strategy(_resolve_strategy(strategy, pop_size))  # type: ignore

    if not verbose:
        opt.cbmanager.unregister(GenerationStartEvent, logging_generation)

    ctx = opt.run()
    return _build_result(ctx)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def minimize(
    func: Callable | Problem,
    algorithm: str | Algorithm | None = _UNSET,  # type: ignore  # sentinel default
    *,
    dim: int | None = None,
    lb=None,
    ub=None,
    n_obj: int = 1,
    direction: np.ndarray | list[str] | None = None,
    surrogate: str | Surrogate | SurrogateManager | None = _UNSET,  # type: ignore
    strategy: str | OptimizationStrategy | None = _UNSET,  # type: ignore
    preset: str | Path | dict | None = None,
    max_fe: int | None = None,
    pop_size: int | None = None,
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
    algorithm : str, Algorithm, or None
        ``'GA'``, ``'PSO'``, or an :class:`Algorithm` instance. If omitted,
        the library's bundled default preset resolves it (currently GA with
        BLX-alpha crossover and uniform mutation).
    dim : int, optional
        Number of design variables. Required when *func* is a callable.
    lb : array-like, optional
        Lower bounds of length *dim*. Required when *func* is a callable.
    ub : array-like, optional
        Upper bounds of length *dim*. Required when *func* is a callable.
    n_obj : int
        Number of objectives. Ignored when *func* is a :class:`Problem`. Default: 1.
    direction : np.ndarray, list[str], or None
        Optimization direction per objective. Each element is ``-1``/``"minimize"``
        (minimize) or ``+1``/``"maximize"`` (maximize). Default: all minimize.
    surrogate : str, Surrogate, SurrogateManager, or None
        ``'rbf'``, a :class:`Surrogate`, or a :class:`SurrogateManager`. If
        omitted, the library's bundled default preset resolves it (currently
        an RBF surrogate with mean-prediction acquisition). Passing ``None``
        explicitly is not supported and raises :class:`ValidationError`.
    strategy : str, OptimizationStrategy, or None
        ``'ib'``, ``'gb'``, ``'ps'``, or an :class:`OptimizationStrategy`. If
        omitted, the library's bundled default preset resolves it (currently
        individual-based).
    preset : str, Path, dict, or None, optional
        A preset (YAML file path or dict) providing default component
        configuration. See :meth:`Optimizer.set_preset`. Components explicitly
        passed via *algorithm*/*surrogate*/*strategy* still take precedence.
    max_fe : int or None
        Maximum true function evaluations. Default: ``200 * dim``.
    pop_size : int or None
        Population size. Default: ``4 * dim``.
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
    >>> result.x, result.f
    """
    direction_arr = _resolve_direction(direction, n_obj, default=-1.0)
    problem = _ensure_problem(func, dim, lb, ub, n_obj, direction_arr)
    return _run(
        problem,
        algorithm,
        surrogate,
        strategy,
        max_fe,
        pop_size,
        seed,
        verbose,
        preset,
    )


def maximize(
    func: Callable | Problem,
    algorithm: str | Algorithm | None = _UNSET,  # type: ignore  # sentinel default
    *,
    dim: int | None = None,
    lb=None,
    ub=None,
    n_obj: int = 1,
    direction: np.ndarray | list[str] | None = None,
    surrogate: str | Surrogate | SurrogateManager | None = _UNSET,  # type: ignore
    strategy: str | OptimizationStrategy | None = _UNSET,  # type: ignore
    preset: str | Path | dict | None = None,
    max_fe: int | None = None,
    pop_size: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> Result:
    """Run surrogate-assisted maximization.

    Identical to :func:`minimize` except that all objectives are maximized
    (``direction = +1``).

    Parameters
    ----------
    func : callable or Problem
        Objective function ``f(x) -> float | array``, or a fully configured
        :class:`Problem` instance.
    algorithm : str, Algorithm, or None
        ``'GA'``, ``'PSO'``, or an :class:`Algorithm` instance. If omitted,
        the library's bundled default preset resolves it (currently GA with
        BLX-alpha crossover and uniform mutation).
    dim : int, optional
        Number of design variables. Required when *func* is a callable.
    lb : array-like, optional
        Lower bounds of length *dim*. Required when *func* is a callable.
    ub : array-like, optional
        Upper bounds of length *dim*. Required when *func* is a callable.
    n_obj : int
        Number of objectives. Ignored when *func* is a :class:`Problem`. Default: 1.
    direction : np.ndarray, list[str], or None
        Optimization direction per objective. Each element is ``-1``/``"minimize"``
        (minimize) or ``+1``/``"maximize"`` (maximize). Default: all maximize.
    surrogate : str, Surrogate, SurrogateManager, or None
        ``'rbf'``, a :class:`Surrogate`, or a :class:`SurrogateManager`. If
        omitted, the library's bundled default preset resolves it (currently
        an RBF surrogate with mean-prediction acquisition). Passing ``None``
        explicitly is not supported and raises :class:`ValidationError`.
    strategy : str, OptimizationStrategy, or None
        ``'ib'``, ``'gb'``, ``'ps'``, or an :class:`OptimizationStrategy`. If
        omitted, the library's bundled default preset resolves it (currently
        individual-based).
    preset : str, Path, dict, or None, optional
        A preset (YAML file path or dict) providing default component
        configuration. See :meth:`Optimizer.set_preset`. Components explicitly
        passed via *algorithm*/*surrogate*/*strategy* still take precedence.
    max_fe : int or None
        Maximum true function evaluations. Default: ``200 * dim``.
    pop_size : int or None
        Population size. Default: ``4 * dim``.
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
    >>> result.x, result.f
    """
    direction_arr = _resolve_direction(direction, n_obj, default=+1.0)
    problem = _ensure_problem(func, dim, lb, ub, n_obj, direction_arr)
    return _run(
        problem,
        algorithm,
        surrogate,
        strategy,
        max_fe,
        pop_size,
        seed,
        verbose,
        preset,
    )
